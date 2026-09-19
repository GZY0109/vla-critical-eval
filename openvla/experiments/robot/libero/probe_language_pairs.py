"""
probe_language_pairs.py

Phase 3 轴一（语言接地反事实测试）的场景/指令配对枚举脚本。CPU-only，不加载 VLA，不占 GPU，可以跟
axis-2 的 GPU 评测同时跑。

libero_spatial 十个任务全部是同一个目标结构："捡起 [某个位置描述] 的黑碗，放到盘子上"——核实过每个
任务的 BDDL 文件，goal 谓词永远是 `(On akita_black_bowl_1 plate_1)`，也就是说目标碗永远是
`akita_black_bowl_1`，陪衬碗永远是 `akita_black_bowl_2`，语言描述换的只是"目标碗初始在哪个位置"这件
事，不是目标结构本身。

这个脚本枚举所有 (scene_task, scene_init_idx) 的陪衬碗（bowl_2）位置，和所有 (instr_task, instr_init_idx)
的目标碗（bowl_1）位置，找出陪衬碗位置落在目标碗典型位置 3cm 以内的组合——复用 scene_task 的这个具体
init_state + instr_task 的语言，就能做反事实测试：策略如果语言接地是对的，应该去抓 scene 里那个陪衬碗
（因为它现在物理位置符合新指令的描述），而不是 scene 原本训练时对应的目标碗。

每个 (scene_task, instr_task) 有序任务对只保留距离最小的一个 init_state 组合（50 个 init_state 大概率
是同一个"槽位"附近的小范围随机扰动，不去重会导致几十个近乎重复的 pair，跑评测时无谓增加 episode 数
却不增加独立信息量）。

Run: cd openvla && python experiments/robot/libero/probe_language_pairs.py
Output: experiments/robot/libero/language_grounding_pairs.json
"""

import json
import os

import numpy as np
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env

DISTANCE_THRESHOLD_M = 0.03
NUM_STEPS_WAIT = 10
TARGET_BOWL = "akita_black_bowl_1"
DISTRACTOR_BOWL = "akita_black_bowl_2"


def _bowl_positions(env):
    """Returns {object_name: xyz} for the two black bowls after they've settled."""
    positions = {}
    for obj in env.env.model.mujoco_objects:
        if obj.name in (TARGET_BOWL, DISTRACTOR_BOWL):
            joint_name = obj.joints[0]
            positions[obj.name] = np.array(env.sim.data.get_joint_qpos(joint_name)[:3])
    return positions


def main():
    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    n_tasks = task_suite.n_tasks
    dummy_action = get_libero_dummy_action("openvla")

    # target_pos[task_id][init_idx] = settled xyz of the target bowl (akita_black_bowl_1)
    # distractor_pos[task_id][init_idx] = settled xyz of the distractor bowl (akita_black_bowl_2)
    target_pos, distractor_pos, descriptions = {}, {}, {}
    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, model_family="openvla")
        descriptions[task_id] = task_description
        target_pos[task_id], distractor_pos[task_id] = {}, {}

        for init_idx in range(len(init_states)):
            env.reset()
            env.set_init_state(init_states[init_idx])
            for _ in range(NUM_STEPS_WAIT):
                env.step(dummy_action)
            pos = _bowl_positions(env)
            target_pos[task_id][init_idx] = pos[TARGET_BOWL]
            distractor_pos[task_id][init_idx] = pos[DISTRACTOR_BOWL]
        env.close()
        print(f"[*] task {task_id} ({task_description}): settled {len(init_states)} init states")

    # For every ordered (scene_task, instr_task) pair, find the (scene_init_idx, instr_init_idx) combo
    # that minimizes distance(scene's distractor bowl, instr_task's target bowl).
    pairs = []
    for scene_task in range(n_tasks):
        for instr_task in range(n_tasks):
            if scene_task == instr_task:
                continue
            best = None
            for scene_init in distractor_pos[scene_task]:
                d_pos = distractor_pos[scene_task][scene_init]
                for instr_init in target_pos[instr_task]:
                    t_pos = target_pos[instr_task][instr_init]
                    dist = float(np.linalg.norm(d_pos - t_pos))
                    if best is None or dist < best["distance_m"]:
                        best = {
                            "scene_task_id": scene_task, "scene_init_idx": scene_init,
                            "instr_task_id": instr_task, "instr_init_idx": instr_init,
                            "distance_m": dist,
                        }
            if best is not None and best["distance_m"] < DISTANCE_THRESHOLD_M:
                best["scene_task_description"] = descriptions[scene_task]
                best["instr_task_description"] = descriptions[instr_task]
                pairs.append(best)

    pairs.sort(key=lambda p: p["distance_m"])
    print(f"[*] found {len(pairs)} pairs under {DISTANCE_THRESHOLD_M * 100:.0f}cm threshold")
    for p in pairs:
        print(f"    scene=task{p['scene_task_id']}(init{p['scene_init_idx']}) + "
              f"instr=task{p['instr_task_id']} -> {p['distance_m']*100:.2f}cm")

    out_path = os.path.join(os.path.dirname(__file__), "language_grounding_pairs.json")
    with open(out_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"[*] wrote {out_path}")


if __name__ == "__main__":
    main()
