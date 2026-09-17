"""
grpo_rollout.py

Rollout collection for GRPO fine-tuning of OpenVLA on LIBERO. Reuses the existing eval infrastructure
(experiments/robot/libero/libero_utils.py, experiments/robot/robot_utils.py, experiments/robot/openvla_utils.py)
wherever possible; only adds the pieces needed to sample actions (instead of greedy-decode them) and capture
per-token log-probs for later GRPO policy-gradient updates.

IMPORTANT (see PROJECT_PLAN.md / the approved Phase 2 plan): `OpenVLAForActionPrediction.generate()` only
supports batch size 1 (hard assertion in modeling_prismatic.py). Rollout collection is therefore a sequential
single-env loop, not a batched/vectorized one -- this mirrors run_libero_eval.py's structure.
"""

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from PIL import Image

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_image, quat2axisangle
from experiments.robot.openvla_utils import DEVICE, crop_and_resize
from experiments.robot.robot_utils import ACTION_DIM, invert_gripper_action, normalize_gripper_action
from prismatic.vla.grpo_utils import Step, Trajectory


def _build_prompt(task_label: str) -> str:
    # Mirrors get_vla_action()'s non-v01 branch in openvla_utils.py -- the only prompt format ever
    # used by the checkpoints this project targets (openvla-7b-finetuned-libero-spatial).
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def _preprocess_image(obs: dict, center_crop: bool) -> Image.Image:
    """Mirrors the image-preprocessing branch of get_vla_action() in openvla_utils.py."""
    image = Image.fromarray(obs["full_image"]).convert("RGB")
    if center_crop:
        batch_size = 1
        crop_scale = 0.9
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = crop_and_resize(image, crop_scale, batch_size)
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
        image = Image.fromarray(image.numpy()).convert("RGB")
    return image


def _decode_action_tokens(vla, action_token_ids: torch.Tensor, unnorm_key: str) -> np.ndarray:
    """Replicates OpenVLAForActionPrediction.predict_action()'s bin-decode + unnormalize block
    (modeling_prismatic.py:520-536), factored out so it can be reused with sampled (not just
    greedily-generated) action tokens."""
    predicted_action_token_ids = action_token_ids.cpu().numpy()
    discretized_actions = vla.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=vla.bin_centers.shape[0] - 1)
    normalized_actions = vla.bin_centers[discretized_actions]

    action_norm_stats = vla.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return actions


def get_vla_action_with_logprobs(vla, processor, obs: dict, task_label: str, unnorm_key: str, center_crop: bool):
    """Samples an action from the VLA (do_sample=True) and returns it along with everything needed to
    recompute this token sequence's log-prob under an updated policy later (GRPO teacher-forced recompute).

    Returns:
        action: np.ndarray (ACTION_DIM,) -- unnormalized continuous action, same format as get_vla_action()
        step: Step -- input_ids/pixel_values/action_token_ids/old_logp for the GRPO update
    """
    image = _preprocess_image(obs, center_crop)
    prompt = _build_prompt(task_label)
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # Mirrors predict_action()'s handling of the trailing empty-string token (29871) -- but unlike
    # predict_action() (inference-only, doesn't care about log-prob consistency across calls), we MUST also
    # extend attention_mask to match, or generate()'s forward pass sees a mask one token short of input_ids,
    # which desyncs it from the fresh full-length mask built in grpo_finetune.py's teacher-forced recompute.
    # (Discovered via the smoke test's new_logp<->old_logp assertion -- see PROJECT_PLAN.md Phase 2.)
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], dtype=input_ids.dtype, device=input_ids.device)), dim=1
        )
        attention_mask = torch.cat(
            (attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)), dim=1
        )

    with torch.no_grad():
        generated = vla.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=inputs["pixel_values"],
            max_new_tokens=ACTION_DIM,
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            return_dict_in_generate=True,
            output_scores=True,
        )

    action_token_ids = generated.sequences[0, -ACTION_DIM:].detach().clone()
    # NOTE: must match top_k=0/top_p=1.0 above -- otherwise `.scores` reflects a truncated distribution
    # (HF's do_sample=True default applies top_k=50), which would silently bias old_logp vs. the untruncated
    # log_softmax computed during the teacher-forced recompute in grpo_finetune.py. See smoke-test step 5.
    old_logp = torch.stack(
        [F.log_softmax(generated.scores[t][0], dim=-1)[action_token_ids[t]] for t in range(ACTION_DIM)]
    ).detach()

    action = _decode_action_tokens(vla, action_token_ids, unnorm_key)

    step = Step(
        input_ids=input_ids[0].detach().cpu(),
        pixel_values=inputs["pixel_values"][0].detach().cpu(),
        action_token_ids=action_token_ids.cpu(),
        old_logp=old_logp.cpu(),
    )
    return action, step


def get_vla_action_with_entropy(vla, processor, obs: dict, task_label: str, unnorm_key: str, center_crop: bool):
    """Greedy-decodes an action (same behavior as get_vla_action() in openvla_utils.py -- used for eval, not
    training rollouts) and additionally returns, for each of the 7 action-token positions, the softmax entropy
    over the 256 action-bin tokens (the last `n_bins` tokens of the vocab -- see ActionTokenizer). Used by
    grpo_eval_libero.py's structured per-episode logging for the SFT-vs-GRPO mechanism-analysis comparison
    (PROJECT_PLAN.md Phase 2/3): does GRPO fine-tuning make the action distribution more peaked (lower entropy)?
    """
    image = _preprocess_image(obs, center_crop)
    prompt = _build_prompt(task_label)
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], dtype=input_ids.dtype, device=input_ids.device)), dim=1
        )
        attention_mask = torch.cat(
            (attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)), dim=1
        )

    with torch.no_grad():
        generated = vla.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=inputs["pixel_values"],
            max_new_tokens=ACTION_DIM,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    action_token_ids = generated.sequences[0, -ACTION_DIM:].detach().clone()
    n_bins = vla.bin_centers.shape[0] + 1  # ActionTokenizer default: 256 bins -> 255 bin_centers
    per_step_entropy = []
    per_step_top1_margin = []
    for t in range(ACTION_DIM):
        bin_logits = generated.scores[t][0][-n_bins:].float()  # last n_bins vocab entries = action-bin tokens
        probs = F.softmax(bin_logits, dim=-1)
        log_probs = F.log_softmax(bin_logits, dim=-1)
        entropy = -(probs * log_probs).sum().item()
        top2 = torch.topk(bin_logits, k=2).values
        per_step_entropy.append(entropy)
        per_step_top1_margin.append((top2[0] - top2[1]).item())

    action = _decode_action_tokens(vla, action_token_ids, unnorm_key)
    return action, per_step_entropy, per_step_top1_margin, action_token_ids.cpu().tolist()


def _max_steps_for_task_suite(task_suite_name: str) -> int:
    # Same table as run_libero_eval.py.
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[task_suite_name]


def collect_one_rollout(
    vla,
    processor,
    env,
    task_id: int,
    task_description: str,
    init_state_idx: int,
    initial_state,
    unnorm_key: str,
    task_suite_name: str,
    center_crop: bool,
    num_steps_wait: int,
    resize_size: int = 224,
) -> Trajectory:
    """Runs one sampled episode and returns a Trajectory with every queried step recorded."""
    env.reset()
    obs = env.set_init_state(initial_state)
    max_steps = _max_steps_for_task_suite(task_suite_name)

    traj = Trajectory(task_id=task_id, task_description=task_description, init_state_idx=init_state_idx, success=False)
    t = 0
    done = False
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
            t += 1
            continue

        img = get_libero_image(obs, resize_size)
        observation = {
            "full_image": img,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }

        action, step = get_vla_action_with_logprobs(vla, processor, observation, task_description, unnorm_key, center_crop)
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        traj.steps.append(step)

        obs, reward, done, info = env.step(action.tolist())
        if done:
            traj.success = True
            break
        t += 1

    return traj


def collect_group_rollouts(
    vla,
    processor,
    env,
    task_id: int,
    task_description: str,
    init_state_idx: int,
    initial_state,
    unnorm_key: str,
    task_suite_name: str,
    center_crop: bool,
    num_steps_wait: int,
    group_size: int,
) -> list:
    """Collects `group_size` independently-sampled rollouts from the same (task, initial_state) -- this is
    the "group" GRPO's advantage is normalized within. Returns list[Trajectory]."""
    return [
        collect_one_rollout(
            vla,
            processor,
            env,
            task_id,
            task_description,
            init_state_idx,
            initial_state,
            unnorm_key,
            task_suite_name,
            center_crop,
            num_steps_wait,
        )
        for _ in range(group_size)
    ]
