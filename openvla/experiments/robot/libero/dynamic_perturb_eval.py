"""
dynamic_perturb_eval.py

Phase 3 轴二：动态场景鲁棒性评测。基于 grpo_eval_libero.py 的评测主循环，额外在 episode 开头一段
固定窗口内持续给场景里所有可动物体（黑碗）注入水平速度（每步重新 set_joint_qvel，抵消摩擦力，
参考 probe_dynamic_object.py 的验证结果），窗口结束后停止注入、让物体在摩擦力下自然停下来——不是
整个 episode 持续扰动，避免"物体被自己推进目标容器"这种跟策略动作无关的混淆结果。

v=0（无扰动）不在这个脚本里跑：直接复用 Phase 2 的
GRPOEVAL-libero_spatial-2026_09_18-12_29_00--sft_baseline_recheck /
GRPOEVAL-libero_spatial-2026_09_18-14_42_40--grpo_real_run_v1_checkpoint 两组 n=120 结果作为衰减
曲线的基线点。

成功判定沿用环境原生 done 信号——这一轴没有换指令、目标物体没变，原生判定仍然有效，不需要代理指标
（跟轴一的语言接地测试不同）。

速度档位校准：用 `calibrate_perturb_velocity.py`（dummy action、无 VLA、只测纯物理）量过，30 步窗口
下 v=0.02/0.05/0.1/0.2 分别对应约 0.2/0.5/1.5/6cm 的纯扰动位移；真实评测里策略会主动伸手，机械臂
本身的接触会再叠加几倍位移（冒烟测试实测 v=0.1 时窗口位移约 7cm），所以正式跑选的档位
（0.05/0.1/0.2）比这次校准数字看起来保守，是因为要给"策略主动干扰"留出空间，不让最高档直接把碗
推出场景。

Usage:
    python experiments/robot/libero/dynamic_perturb_eval.py \
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --velocities 0.05,0.1,0.2 --num_trials_per_task 5 --run_id_note sft_dynamic_perturb

    python experiments/robot/libero/dynamic_perturb_eval.py \
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --adapter_path "runs/grpo/.../adapter_latest" \
        --velocities 0.05,0.1,0.2 --num_trials_per_task 5 --run_id_note grpo_dynamic_perturb
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
class DynamicPerturbConfig:
    # fmt: off
    pretrained_checkpoint: Union[str, Path] = ""
    adapter_path: Optional[Union[str, Path]] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 5

    velocities: str = "0.05,0.1,0.2"   # comma-separated horizontal speeds (m/s) to sweep, one full pass per value
    perturb_window_steps: int = 30     # only re-inject velocity for this many policy-control steps, then let friction settle it

    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    seed: int = 7
    # fmt: on


def _max_steps_for_task_suite(task_suite_name: str) -> int:
    return {
        "libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400,
    }[task_suite_name]


def _movable_joints(env):
    """Returns [(object_name, joint_name), ...] for every scene object with a *free* joint (7-dof qpos:
    xyz + quaternion) -- the black bowls in libero_spatial tasks. `obj.joints` also includes non-free
    joints (e.g. a cabinet door's 1-dof hinge), whose qpos is a scalar and isn't a movable prop we can
    inject a translational velocity into, so filter by qpos shape rather than just `if obj.joints`."""
    joints = []
    for obj in env.env.model.mujoco_objects:
        for joint_name in obj.joints:
            qpos = np.atleast_1d(env.sim.data.get_joint_qpos(joint_name))
            if qpos.shape[0] == 7:
                joints.append((obj.name, joint_name))
                break
    return joints


@draccus.wrap()
def dynamic_perturb_eval(cfg: DynamicPerturbConfig) -> None:
    assert cfg.pretrained_checkpoint, "cfg.pretrained_checkpoint must not be empty!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    velocities = [float(v) for v in cfg.velocities.split(",") if v.strip()]
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

    run_id = f"DYNPERTURB-{cfg.task_suite_name}-{DATE_TIME}"
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
    max_steps = _max_steps_for_task_suite(cfg.task_suite_name)

    for velocity in velocities:
        print(f"\n[*] === velocity = {velocity} m/s ===")
        log_file.write(f"\n=== velocity = {velocity} m/s ===\n")
        total_episodes, total_successes = 0, 0

        for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, "openvla", resolution=256)
            movable = _movable_joints(env)

            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                start_positions = {name: np.array(env.sim.data.get_joint_qpos(joint)[:3]) for name, joint in movable}

                t = 0
                done = False
                replay_images = []
                per_step_entropy = []
                perturb_end_positions = None
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

                    action, entropy, _, _ = get_vla_action_with_entropy(
                        vla, processor, observation, task_description, unnorm_key, cfg.center_crop
                    )
                    per_step_entropy.append(entropy)
                    action = normalize_gripper_action(action, binarize=True)
                    action = invert_gripper_action(action)

                    perturb_step_idx = t - cfg.num_steps_wait
                    if perturb_step_idx < cfg.perturb_window_steps:
                        for _, joint_name in movable:
                            env.sim.data.set_joint_qvel(joint_name, [velocity, 0.0, 0.0, 0.0, 0.0, 0.0])
                        env.sim.forward()
                    if perturb_step_idx == cfg.perturb_window_steps - 1:
                        # Snapshot positions right as the injection window closes, before any further
                        # steps -- isolates perturbation-caused displacement from displacement the robot
                        # itself later causes by legitimately manipulating the object (which can be much
                        # larger, e.g. carrying a bowl 20-30cm to its target container).
                        perturb_end_positions = {
                            name: np.array(env.sim.data.get_joint_qpos(joint)[:3]) for name, joint in movable
                        }

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                task_episodes += 1
                total_episodes += 1
                end_positions = {name: np.array(env.sim.data.get_joint_qpos(joint)[:3]) for name, joint in movable}
                if perturb_end_positions is None:  # episode ended before the perturbation window even closed
                    perturb_end_positions = end_positions
                displacement = {name: float(np.linalg.norm(end_positions[name] - start_positions[name])) for name in end_positions}
                perturb_displacement = {
                    name: float(np.linalg.norm(perturb_end_positions[name] - start_positions[name])) for name in end_positions
                }

                save_rollout_video(
                    replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
                )

                jsonl_file.write(json.dumps({
                    "velocity": velocity, "task_id": task_id, "task_description": task_description,
                    "init_state_idx": episode_idx, "success": bool(done), "num_steps": len(per_step_entropy),
                    "object_displacement_m": displacement,
                    "perturbation_window_displacement_m": perturb_displacement,
                    "mean_entropy": float(np.mean(per_step_entropy)) if per_step_entropy else None,
                }) + "\n")
                jsonl_file.flush()

                print(f"[v={velocity}] Success: {done} | episodes so far: {total_episodes} | "
                      f"successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                log_file.write(f"Success: {done}\n# episodes completed so far: {total_episodes}\n"
                                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
                log_file.flush()

            print(f"[v={velocity}] task {task_id} success rate: {float(task_successes) / float(task_episodes)}")
            log_file.write(f"Task {task_id} success rate: {float(task_successes) / float(task_episodes)}\n")
            log_file.flush()
            env.close()

        print(f"[*] velocity={velocity} final success rate: {total_successes / total_episodes:.4f} "
              f"({total_successes}/{total_episodes})")
        log_file.write(f"velocity={velocity} final success rate: {total_successes / total_episodes:.4f} "
                        f"({total_successes}/{total_episodes})\n")
        log_file.flush()

    log_file.close()
    jsonl_file.close()
    print("[*] Dynamic perturbation eval finished for all velocity levels.")


if __name__ == "__main__":
    dynamic_perturb_eval()
