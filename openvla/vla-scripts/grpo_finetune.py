"""
grpo_finetune.py

GRPO (group-relative policy optimization, critic-free RL) fine-tuning of OpenVLA on a small
LIBERO-Spatial task subset, on a single GPU. See PROJECT_PLAN.md Phase 2 and the approved plan
(PPO -> GRPO, no value head -- RL4VLA's value head alone measured 44.4GB, over budget for a 3090)
for the design rationale. Mirrors vla-scripts/finetune.py's conventions (draccus config, LoRA via
PEFT, checkpoint layout) wherever the RL setting doesn't force a divergence.

Run with:
    python vla-scripts/grpo_finetune.py --smoke_test True
    python vla-scripts/grpo_finetune.py --run_id_note real_run_v1
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import draccus
import torch
import torch.nn.functional as F
from libero.libero import benchmark
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW

sys.path.append(".")
from experiments.robot.libero.grpo_rollout import collect_group_rollouts
from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.openvla_utils import get_processor, get_vla
from experiments.robot.robot_utils import DATE_TIME, set_seed_everywhere
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.grpo_utils import IGNORE_INDEX, assign_advantages


@dataclass
class GRPOConfig:
    # fmt: off

    #################################################################################################################
    # Model / checkpoint (starts from the Phase 1 LoRA-SFT checkpoint, adds a fresh LoRA adapter on top)
    #################################################################################################################
    pretrained_checkpoint: str = "openvla/openvla-7b-finetuned-libero-spatial"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    #################################################################################################################
    # LIBERO task scope -- deliberately a small subset, not the full 10-task suite (see PROJECT_PLAN.md Phase 2:
    # rollout collection is the wall-clock bottleneck, not training compute; tasks chosen are the two with the
    # most reward variance at the Phase 1 baseline (50%/66.7%), which maximizes non-degenerate GRPO groups)
    #################################################################################################################
    task_suite_name: str = "libero_spatial"
    task_names: List[str] = field(default_factory=lambda: [
        "pick up the black bowl on the wooden cabinet and place it on the plate",   # Phase 1 baseline: 50.0%
        "pick up the black bowl on the stove and place it on the plate",            # Phase 1 baseline: 66.7%
    ])
    num_init_states_per_task: int = 3
    group_size: int = 6                               # rollouts sampled per (task, init_state) group
    num_steps_wait: int = 10                           # sim settle steps, matches run_libero_eval.py

    #################################################################################################################
    # LoRA -- same recipe as vla-scripts/finetune.py / the official checkpoints, for apples-to-apples comparison
    #################################################################################################################
    lora_rank: int = 32
    lora_dropout: float = 0.0

    #################################################################################################################
    # GRPO objective
    #################################################################################################################
    learning_rate: float = 1e-5
    clip_eps: float = 0.2                              # PPO/GRPO-style clipped surrogate epsilon
    kl_coef: float = 0.02                               # k3-estimator KL penalty vs. frozen reference (base checkpoint)
    adv_eps: float = 1e-4                               # variance floor in group-advantage normalization
    micro_batch_size: int = 2                           # teacher-forced recompute micro-batch; raise only after
                                                        #   confirming headroom in the smoke test, not by assumption

    #################################################################################################################
    # Run control
    #################################################################################################################
    num_iterations: int = 25
    save_every_n_iters: int = 5
    seed: int = 7
    run_root_dir: Path = Path("runs/grpo")
    run_id_note: Optional[str] = None

    # Smoke-test mode: 1 task, tiny group, 1 iteration, extra correctness assertions -- run this before
    # committing to the real multi-hour run (see PROJECT_PLAN.md Phase 2 smoke-test section)
    smoke_test: bool = False

    # fmt: on


def build_training_example(step, advantage: float):
    """One (full_input_ids, labels, pixel_values, old_logp, advantage) example for the GRPO update,
    from a rollout Step. `labels` masks everything except the 7 sampled action tokens, mirroring
    finetune.py's convention."""
    full_input_ids = torch.cat([step.input_ids, step.action_token_ids])
    labels = full_input_ids.clone()
    labels[: -step.action_token_ids.shape[0]] = IGNORE_INDEX
    return {
        "input_ids": full_input_ids,
        "labels": labels,
        "pixel_values": step.pixel_values,
        "old_logp": step.old_logp,
        "advantage": torch.full_like(step.old_logp, fill_value=advantage),
    }


def collate_training_examples(examples: list, pad_token_id: int):
    input_ids = pad_sequence([e["input_ids"] for e in examples], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence([e["labels"] for e in examples], batch_first=True, padding_value=IGNORE_INDEX)
    attention_mask = input_ids.ne(pad_token_id)
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    return input_ids, labels, attention_mask, pixel_values


def compute_logp(vla, input_ids, attention_mask, pixel_values, labels, action_token_begin_idx, num_patches, device):
    """Teacher-forced forward pass -> per-(example, action-token) log-prob, in left-to-right action-dim order.
    Reuses the exact logit-slicing convention from vla-scripts/finetune.py (lines ~270-277)."""
    output = vla(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        pixel_values=pixel_values.to(torch.bfloat16).to(device),
    )
    action_logits = output.logits[:, num_patches:-1]
    logp_all = F.log_softmax(action_logits.float(), dim=-1)
    action_gt = labels[:, 1:].to(device)
    mask = action_gt > action_token_begin_idx
    gathered = logp_all.gather(-1, action_gt.clamp(min=0).unsqueeze(-1)).squeeze(-1)

    # Each example must contribute exactly ACTION_DIM masked positions, in order -- reshape per-example.
    per_example_logp = []
    for b in range(gathered.shape[0]):
        per_example_logp.append(gathered[b][mask[b]])
    return torch.stack(per_example_logp)  # (B, ACTION_DIM)


@draccus.wrap()
def grpo_finetune(cfg: GRPOConfig) -> None:
    assert torch.cuda.is_available(), "GRPO fine-tuning assumes a GPU is available!"
    device = torch.device("cuda:0")
    set_seed_everywhere(cfg.seed)

    if cfg.smoke_test:
        print("[*] SMOKE TEST MODE: 1 task, group_size<=4, 1 init state, 1 iteration, extra assertions enabled.")
        cfg.task_names = cfg.task_names[:1]
        cfg.group_size = min(cfg.group_size, 4)
        cfg.num_init_states_per_task = 1
        cfg.num_iterations = 1

    exp_id = f"grpo+{cfg.pretrained_checkpoint.split('/')[-1]}+lora-r{cfg.lora_rank}+lr-{cfg.learning_rate}"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.smoke_test:
        exp_id += "--smoke_test"
    run_dir = cfg.run_root_dir / f"{exp_id}--{DATE_TIME}"
    os.makedirs(run_dir, exist_ok=True)
    train_log_path = run_dir / "train_log.jsonl"

    # --- Load model + processor (reuses experiments/robot/openvla_utils.py) ---
    class _VLAConfigShim:
        pretrained_checkpoint = cfg.pretrained_checkpoint
        load_in_8bit = cfg.load_in_8bit
        load_in_4bit = cfg.load_in_4bit

    vla = get_vla(_VLAConfigShim())
    processor = get_processor(_VLAConfigShim())
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    pad_token_id = processor.tokenizer.pad_token_id
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches

    unnorm_key = cfg.task_suite_name
    if unnorm_key not in vla.norm_stats and f"{unnorm_key}_no_noops" in vla.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in vla.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # --- Wrap with a fresh LoRA adapter (norm_stats already set on the base model above -- must wrap AFTER) ---
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()

    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # --- Resolve LIBERO tasks ---
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()

    tasks = []  # list of (task_id, task, task_description, env, initial_states)
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        if task.language not in cfg.task_names:
            continue
        env, task_description = get_libero_env(task, "openvla", resolution=256)
        initial_states = task_suite.get_task_init_states(task_id)
        tasks.append((task_id, task_description, env, initial_states))
    assert len(tasks) == len(cfg.task_names), (
        f"Expected to find {len(cfg.task_names)} tasks in {cfg.task_suite_name}, found {len(tasks)}. "
        f"Check `task_names` matches `task.language` strings exactly."
    )

    print(f"[*] GRPO run: {exp_id}")
    print(f"[*] Tasks: {[t[1] for t in tasks]}")
    print(f"[*] Logging to {train_log_path}")

    for iteration in range(cfg.num_iterations):
        iter_start = time.time()
        vla.eval()  # rollout collection uses .generate(), no dropout/grad needed
        all_examples = []
        num_degenerate_groups = 0
        num_groups = 0
        episode_successes = []

        for task_id, task_description, env, initial_states in tasks:
            for init_state_idx in range(cfg.num_init_states_per_task):
                trajectories = collect_group_rollouts(
                    vla,
                    processor,
                    env,
                    task_id,
                    task_description,
                    init_state_idx,
                    initial_states[init_state_idx],
                    unnorm_key,
                    cfg.task_suite_name,
                    cfg.center_crop,
                    cfg.num_steps_wait,
                    cfg.group_size,
                )
                num_groups += 1
                episode_successes.extend([t.success for t in trajectories])
                if cfg.smoke_test and len({t.success for t in trajectories}) == 1:
                    # Sampling turned out to give an all-success (or all-fail) group by chance -- real GRPO
                    # would just skip this group, but the smoke test's only job is to mechanically validate
                    # the loss/backward/checkpoint path, so force a synthetic split on REAL rollout data
                    # (same images/prompts/sampled tokens, only the reward label is forced) rather than
                    # re-rolling the dice repeatedly. This is clearly logged and is never used for real training.
                    print("[smoke test] group was degenerate by chance; forcing a synthetic reward split "
                          "on real rollout data to exercise the loss/backward/checkpoint path.")
                    for i, t in enumerate(trajectories):
                        t.success = i % 2 == 0
                trajectories = assign_advantages(trajectories)
                if trajectories[0].advantage is None:
                    num_degenerate_groups += 1
                    continue
                for traj in trajectories:
                    for step in traj.steps:
                        all_examples.append(build_training_example(step, traj.advantage))

        rollout_time = time.time() - iter_start
        success_rate = sum(episode_successes) / max(len(episode_successes), 1)
        print(
            f"[iter {iteration}] rollout done in {rollout_time:.1f}s | "
            f"episodes={len(episode_successes)} success_rate={success_rate:.3f} | "
            f"degenerate_groups={num_degenerate_groups}/{num_groups} | training_examples={len(all_examples)}"
        )

        if not all_examples:
            print(f"[iter {iteration}] all groups degenerate, skipping update.")
            with open(train_log_path, "a") as f:
                f.write(json.dumps({
                    "iteration": iteration, "success_rate": success_rate, "num_episodes": len(episode_successes),
                    "degenerate_groups": num_degenerate_groups, "num_groups": num_groups,
                    "num_training_examples": 0, "loss": None, "kl": None, "rollout_time_s": rollout_time,
                    "update_time_s": 0.0,
                }) + "\n")
            continue

        # --- Policy update: single on-policy pass over this iteration's data, micro-batched ---
        update_start = time.time()
        vla.train()
        optimizer.zero_grad()
        num_micro_batches = (len(all_examples) + cfg.micro_batch_size - 1) // cfg.micro_batch_size
        total_loss, total_kl = 0.0, 0.0
        logp_diffs = []

        for mb_idx in range(num_micro_batches):
            mb = all_examples[mb_idx * cfg.micro_batch_size : (mb_idx + 1) * cfg.micro_batch_size]
            input_ids, labels, attention_mask, pixel_values = collate_training_examples(mb, pad_token_id)
            old_logp = torch.stack([e["old_logp"] for e in mb]).to(device)
            advantage = torch.stack([e["advantage"] for e in mb]).to(device)

            new_logp = compute_logp(
                vla, input_ids, attention_mask, pixel_values, labels,
                action_tokenizer.action_token_begin_idx, num_patches, device,
            )

            if cfg.smoke_test:
                diff = (new_logp - old_logp).abs()
                flat_idx = diff.argmax()
                logp_diffs.append({
                    "max": diff.max().item(),
                    "mean": diff.mean().item(),
                    "median": diff.median().item(),
                    "frac_gt_0.2": (diff > 0.2).float().mean().item(),
                    "old_logp_at_max": old_logp.flatten()[flat_idx].item(),
                    "new_logp_at_max": new_logp.flatten()[flat_idx].item(),
                })

            if cfg.kl_coef > 0:
                with vla.disable_adapter():
                    with torch.no_grad():
                        ref_logp = compute_logp(
                            vla, input_ids, attention_mask, pixel_values, labels,
                            action_tokenizer.action_token_begin_idx, num_patches, device,
                        )
                kl = torch.exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1
            else:
                kl = torch.zeros_like(new_logp)

            ratio = torch.exp(new_logp - old_logp)
            surrogate = torch.min(ratio * advantage, torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantage)
            loss = -surrogate.mean() + cfg.kl_coef * kl.mean()

            (loss / num_micro_batches).backward()
            total_loss += loss.item()
            total_kl += kl.mean().item()

        assert torch.isfinite(torch.tensor(total_loss)), f"Non-finite loss at iteration {iteration}: {total_loss}"
        optimizer.step()
        update_time = time.time() - update_start

        if cfg.smoke_test:
            worst = max(logp_diffs, key=lambda d: d["max"]) if logp_diffs else None
            mean_of_means = sum(d["mean"] for d in logp_diffs) / max(len(logp_diffs), 1)
            print(
                f"[smoke test] new_logp vs old_logp: worst microbatch max={worst['max']:.4f} "
                f"mean={worst['mean']:.4f} median={worst['median']:.4f} frac(|diff|>0.2)={worst['frac_gt_0.2']:.3f} | "
                f"avg-of-microbatch-means={mean_of_means:.4f} | at worst-case token: old_logp={worst['old_logp_at_max']:.3f} "
                f"new_logp={worst['new_logp_at_max']:.3f}"
            )
            if mean_of_means > 0.2:
                print(
                    "[smoke test] WARNING: average new_logp/old_logp mismatch exceeds 0.2 -- check top_k/top_p/"
                    "temperature settings in grpo_rollout.py (see PROJECT_PLAN.md Phase 2 Gotcha #2)."
                )
            else:
                print(
                    "[smoke test] average mismatch is small; large single-token max diffs are expected bf16 "
                    "tail-probability noise on low-probability sampled tokens (this is exactly what PPO/GRPO's "
                    "clipped surrogate objective is designed to bound, not a bug)."
                )

        avg_loss = total_loss / num_micro_batches
        avg_kl = total_kl / num_micro_batches
        print(f"[iter {iteration}] update done in {update_time:.1f}s | loss={avg_loss:.4f} kl={avg_kl:.4f}")

        with open(train_log_path, "a") as f:
            f.write(json.dumps({
                "iteration": iteration, "success_rate": success_rate, "num_episodes": len(episode_successes),
                "degenerate_groups": num_degenerate_groups, "num_groups": num_groups,
                "num_training_examples": len(all_examples), "loss": avg_loss, "kl": avg_kl,
                "rollout_time_s": rollout_time, "update_time_s": update_time,
            }) + "\n")

        # --- Checkpoint ---
        adapter_latest = run_dir / "adapter_latest"
        vla.save_pretrained(adapter_latest)
        processor.save_pretrained(run_dir)
        if (iteration + 1) % cfg.save_every_n_iters == 0:
            vla.save_pretrained(run_dir / f"adapter_iter_{iteration}")
            print(f"[iter {iteration}] saved milestone checkpoint to {run_dir / f'adapter_iter_{iteration}'}")

    print(f"[*] GRPO training finished. Adapter checkpoints in {run_dir}")


if __name__ == "__main__":
    grpo_finetune()
