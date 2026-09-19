"""Phase 3 探索脚本：验证能不能在 LIBERO episode 中途直接注入物体速度/位置，
让目标物体动起来（AHEAD 风格的动态场景鲁棒性诊断的前置可行性验证）。

不跑 OpenVLA，只测 LIBERO/robosuite/mujoco 环境本身，不需要 GPU 推理，
可以在 GRPO 训练进程占满显存的同时跑。

跑法：cd openvla && python experiments/robot/libero/probe_dynamic_object.py
"""

import numpy as np
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_env

task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = task_suite.get_task(0)
env, task_description = get_libero_env(task, model_family="openvla")

mujoco_objects = env.env.model.mujoco_objects
print(f"[*] task: {task_description}")
print(f"[*] mujoco objects: {[o.name for o in mujoco_objects]}")

init_states = task_suite.get_task_init_states(0)
obs = env.set_init_state(init_states[0])

dummy_action = [0, 0, 0, 0, 0, 0, -1]
for _ in range(10):
    obs, reward, done, info = env.step(dummy_action)

target_obj = mujoco_objects[0]
target_obj_name = target_obj.name
joint_name = target_obj.joints[0]
print(f"[*] target object: {target_obj_name}, joint: {joint_name}")

qpos_before = np.array(env.sim.data.get_joint_qpos(joint_name))
print(f"[*] qpos before inject: {qpos_before}")

# 注入一个水平方向的匀速运动（vx=0.3 m/s），模拟 AHEAD 里的传送带/物体平移场景
env.sim.data.set_joint_qvel(joint_name, [0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
env.sim.forward()

positions = [qpos_before[:3].copy()]
for step in range(20):
    obs, reward, done, info = env.step(dummy_action)
    qpos = np.array(env.sim.data.get_joint_qpos(joint_name))
    positions.append(qpos[:3].copy())

positions = np.array(positions)
print(f"[*] x position over 20 steps after injecting vx=0.3: {positions[:, 0]}")
displacement = positions[-1, 0] - positions[0, 0]
print(f"[*] total x displacement: {displacement:.4f} m")

if abs(displacement) > 0.01:
    print("[RESULT] 物体确实动了，直接改 sim qvel 注入速度的方案可行。")
else:
    print("[RESULT] 物体没有明显位移，方案可能被 robosuite 内部逻辑覆盖，需要进一步排查。")

env.close()
