"""
grpo_eval_libero.py

Copy of run_libero_eval.py extended with structured per-episode JSONL logging (success, episode length,
per-step action-token entropy/top1-margin) needed for the Phase 2/3 SFT-vs-GRPO mechanism-analysis
comparison. Phase 1's plain-text log (eval_phase1_baseline.log) didn't preserve logits, so it can't be
used retroactively for entropy comparison -- both the SFT baseline and the GRPO checkpoint need to be
(re-)evaluated with this script on the same task/trial scope for an apples-to-apples comparison. The
Phase 1 headline number (82.5%, n=120, already committed) is unaffected; this is additional instrumentation,
not a re-litigation of that result.

Usage:
    # Evaluate a plain (merged) checkpoint, e.g. the Phase 1 SFT baseline, with the new instrumentation:
    python experiments/robot/libero/grpo_eval_libero.py \
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial --center_crop True --num_trials_per_task 12

    # Evaluate a GRPO LoRA adapter on top of that same base checkpoint:
    python experiments/robot/libero/grpo_eval_libero.py \
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --adapter_path runs/grpo/<exp_id>/adapter_latest \
        --task_suite_name libero_spatial --center_crop True --num_trials_per_task 12
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from peft import PeftModel

sys.path.append(".")
from experiments.robot.libero.grpo_rollout import get_vla_action_with_entropy
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image, quat2axisangle, save_rollout_video
from experiments.robot.openvla_utils import get_processor, get_vla
from experiments.robot.robot_utils import DATE_TIME, invert_gripper_action, normalize_gripper_action, set_seed_everywhere


@dataclass
class GRPOEvalConfig:
    # fmt: off
    pretrained_checkpoint: Union[str, Path] = ""     # Base (merged) checkpoint path
    adapter_path: Optional[Union[str, Path]] = None  # Optional LoRA adapter to load on top (e.g. a GRPO checkpoint)
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    seed: int = 7
    # fmt: on


def _max_steps_for_task_suite(task_suite_name: str) -> int:
    return {
        "libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400,
    }[task_suite_name]


@draccus.wrap()
def grpo_eval_libero(cfg: GRPOEvalConfig) -> None:
    assert cfg.pretrained_checkpoint, "cfg.pretrained_checkpoint must not be empty!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    set_seed_everywhere(cfg.seed)
    unnorm_key = cfg.task_suite_name

    vla = get_vla(cfg)
    processor = get_processor(cfg)
    if cfg.adapter_path is not None:
        print(f"[*] Loading GRPO LoRA adapter from {cfg.adapter_path}")
        vla = PeftModel.from_pretrained(vla, cfg.adapter_path)
        vla.eval()

    if unnorm_key not in vla.norm_stats and f"{unnorm_key}_no_noops" in vla.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in vla.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    run_id = f"GRPOEVAL-{cfg.task_suite_name}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    txt_log_path = os.path.join(cfg.local_log_dir, run_id + ".txt")
    jsonl_log_path = os.path.join(cfg.local_log_dir, run_id + ".jsonl")
    log_file = open(txt_log_path, "w")
    jsonl_file = open(jsonl_log_path, "w")
    print(f"Logging to {txt_log_path} and {jsonl_log_path}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    resize_size = 224

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, "openvla", resolution=256)
        max_steps = _max_steps_for_task_suite(cfg.task_suite_name)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            done = False
            replay_images = []
            per_step_entropy, per_step_top1_margin, per_step_action_token_ids = [], [], []
            while t < max_steps + cfg.num_steps_wait:
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
                    t += 1
                    continue

                img = get_libero_image(obs, resize_size)
                replay_images.append(img)
                observation = {
                    "full_image": img,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                }

                action, entropy, top1_margin, action_token_ids = get_vla_action_with_entropy(
                    vla, processor, observation, task_description, unnorm_key, cfg.center_crop
                )
                per_step_entropy.append(entropy)
                per_step_top1_margin.append(top1_margin)
                per_step_action_token_ids.append(action_token_ids)

                action = normalize_gripper_action(action, binarize=True)
                action = invert_gripper_action(action)

                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            task_episodes += 1
            total_episodes += 1
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            jsonl_file.write(json.dumps({
                "task_id": task_id, "task_description": task_description, "init_state_idx": episode_idx,
                "success": bool(done), "num_steps": len(per_step_entropy),
                "per_step_entropy": per_step_entropy, "per_step_top1_margin": per_step_top1_margin,
                "per_step_action_token_ids": per_step_action_token_ids,
            }) + "\n")
            jsonl_file.flush()

            print(f"Success: {done} | # episodes so far: {total_episodes} | # successes: {total_successes} "
                  f"({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n# episodes completed so far: {total_episodes}\n"
                            f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

    log_file.close()
    jsonl_file.close()
    print(f"[*] Final success rate: {total_successes / total_episodes:.4f} ({total_successes}/{total_episodes})")


if __name__ == "__main__":
    grpo_eval_libero()
