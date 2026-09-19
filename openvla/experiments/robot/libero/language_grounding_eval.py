"""
language_grounding_eval.py

Phase 3 轴一：语言接地反事实测试。加载 probe_language_pairs.py 产出的 pairs（同场景/换指令的
scene+instruction 配对，陪衬碗位置跟新指令的目标碗典型位置重合 <3cm），对每个 pair：用 scene_task
的具体 init_state 摆场景，但把喂给 VLA 的语言换成 instr_task 的指令，跑一个 episode。

环境原生 success 判定在这里没有意义（绑死在 scene_task 自己的 goal `(On akita_black_bowl_1 plate_1)`
上，跟新指令描述的目标碗是不是同一个无关）——仍然记录下来作为参考信息（比如能看出策略是不是"完全没
理会新指令，还是习惯性去抓 scene 原本训练时的那个目标碗"），但不作为主判据。

代理指标：整个 episode 里，逐步记录末端执行器（eef）到两个碗（akita_black_bowl_1 / _2）的欧氏距离，
取各自的最小值；"策略瞄准的碗" = 两个最小距离里更小的那个对应的碗。因为 pairing 是拿 scene 的陪衬碗
（akita_black_bowl_2）位置去匹配 instr_task 的目标碗典型位置，所以"正确"定义为策略瞄准
akita_black_bowl_2（陪衬碗，但物理位置符合新指令）而不是 akita_black_bowl_1（scene 本来的目标碗，
位置对应 scene_task 的旧指令）。

Usage:
    python experiments/robot/libero/language_grounding_eval.py \
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --run_id_note sft_language_grounding

    python experiments/robot/libero/language_grounding_eval.py \
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --adapter_path "runs/grpo/.../adapter_latest" \
        --run_id_note grpo_language_grounding
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

TARGET_BOWL = "akita_black_bowl_1"      # scene_task's own habitual target -- "wrong" answer under the new instruction
DISTRACTOR_BOWL = "akita_black_bowl_2"  # physically sits where instr_task's target usually is -- "correct" answer
MAX_STEPS = 220  # libero_spatial


@dataclass
class LanguageGroundingConfig:
    # fmt: off
    pretrained_checkpoint: Union[str, Path] = ""
    adapter_path: Optional[Union[str, Path]] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    pairs_path: str = "experiments/robot/libero/language_grounding_pairs.json"

    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    seed: int = 7
    # fmt: on


def _bowl_positions(env):
    positions = {}
    for obj in env.env.model.mujoco_objects:
        if obj.name in (TARGET_BOWL, DISTRACTOR_BOWL):
            joint_name = obj.joints[0]
            positions[obj.name] = np.array(env.sim.data.get_joint_qpos(joint_name)[:3])
    return positions


@draccus.wrap()
def language_grounding_eval(cfg: LanguageGroundingConfig) -> None:
    assert cfg.pretrained_checkpoint, "cfg.pretrained_checkpoint must not be empty!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    set_seed_everywhere(cfg.seed)
    unnorm_key = cfg.task_suite_name

    with open(cfg.pairs_path) as f:
        pairs = json.load(f)
    print(f"[*] loaded {len(pairs)} scene/instruction pairs from {cfg.pairs_path}")

    vla = get_vla(cfg)
    processor = get_processor(cfg)
    if cfg.adapter_path is not None:
        print(f"[*] Loading GRPO LoRA adapter from {cfg.adapter_path}")
        vla = PeftModel.from_pretrained(vla, cfg.adapter_path)
        vla.eval()

    if unnorm_key not in vla.norm_stats and f"{unnorm_key}_no_noops" in vla.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in vla.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    run_id = f"LANGGROUND-{cfg.task_suite_name}-{DATE_TIME}"
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

    n_correct, n_total = 0, 0
    for pair in tqdm.tqdm(pairs):
        scene_task = task_suite.get_task(pair["scene_task_id"])
        instr_task = task_suite.get_task(pair["instr_task_id"])
        init_states = task_suite.get_task_init_states(pair["scene_task_id"])
        env, scene_description = get_libero_env(scene_task, "openvla", resolution=256)
        instr_description = instr_task.language

        env.reset()
        obs = env.set_init_state(init_states[pair["scene_init_idx"]])

        t = 0
        done = False
        replay_images = []
        dist_to_target, dist_to_distractor = [], []
        while t < MAX_STEPS + cfg.num_steps_wait:
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

            # NOTE: fed the borrowed instr_description, not scene_description -- this is the whole point.
            action, _, _, _ = get_vla_action_with_entropy(
                vla, processor, observation, instr_description, unnorm_key, cfg.center_crop
            )
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)

            bowls = _bowl_positions(env)
            eef_pos = obs["robot0_eef_pos"]
            dist_to_target.append(float(np.linalg.norm(eef_pos - bowls[TARGET_BOWL])))
            dist_to_distractor.append(float(np.linalg.norm(eef_pos - bowls[DISTRACTOR_BOWL])))

            obs, reward, done, info = env.step(action.tolist())
            if done:
                break
            t += 1

        min_dist_target = min(dist_to_target) if dist_to_target else float("inf")
        min_dist_distractor = min(dist_to_distractor) if dist_to_distractor else float("inf")
        attended_bowl = DISTRACTOR_BOWL if min_dist_distractor < min_dist_target else TARGET_BOWL
        proxy_correct = attended_bowl == DISTRACTOR_BOWL

        n_total += 1
        n_correct += int(proxy_correct)

        save_rollout_video(
            replay_images, n_total, success=proxy_correct, task_description=instr_description, log_file=log_file
        )

        jsonl_file.write(json.dumps({
            "scene_task_id": pair["scene_task_id"], "scene_init_idx": pair["scene_init_idx"],
            "instr_task_id": pair["instr_task_id"], "pair_distance_m": pair["distance_m"],
            "scene_description": scene_description, "instr_description": instr_description,
            "native_env_success": bool(done), "min_dist_to_target_bowl_m": min_dist_target,
            "min_dist_to_distractor_bowl_m": min_dist_distractor, "attended_bowl": attended_bowl,
            "proxy_correct": proxy_correct, "num_steps": len(dist_to_target),
        }) + "\n")
        jsonl_file.flush()

        print(f"[pair {n_total}/{len(pairs)}] scene=task{pair['scene_task_id']} instr=task{pair['instr_task_id']} "
              f"-> attended={attended_bowl} correct={proxy_correct} | running acc: {n_correct}/{n_total} "
              f"({n_correct / n_total * 100:.1f}%)")
        log_file.write(f"proxy_correct: {proxy_correct}\nrunning acc: {n_correct}/{n_total} "
                        f"({n_correct / n_total * 100:.1f}%)\n")
        log_file.flush()
        env.close()

    log_file.close()
    jsonl_file.close()
    print(f"[*] Final language-grounding proxy accuracy: {n_correct / n_total:.4f} ({n_correct}/{n_total})")


if __name__ == "__main__":
    language_grounding_eval()
