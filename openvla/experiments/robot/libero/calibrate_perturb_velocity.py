"""Fast CPU-only calibration for dynamic_perturb_eval.py's velocity injection: no VLA, dummy no-op
gripper action throughout (isolates perturbation-caused displacement from anything a real policy would
add), sweeps candidate velocities over a 30-step window on task 0's initial state, perturbing all movable
(free-joint) objects at once the same way the real eval script does.

Run: cd openvla && python experiments/robot/libero/calibrate_perturb_velocity.py
"""

import numpy as np
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env

task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = task_suite.get_task(0)
init_states = task_suite.get_task_init_states(0)

for velocity in [0.0, 0.02, 0.05, 0.1, 0.2]:
    env, task_description = get_libero_env(task, model_family="openvla")
    env.reset()
    env.set_init_state(init_states[0])

    dummy_action = get_libero_dummy_action("openvla")
    for _ in range(10):
        env.step(dummy_action)

    movable = []
    for obj in env.env.model.mujoco_objects:
        for joint_name in obj.joints:
            qpos = np.atleast_1d(env.sim.data.get_joint_qpos(joint_name))
            if qpos.shape[0] == 7:
                movable.append((obj.name, joint_name))
                break

    start = {name: np.array(env.sim.data.get_joint_qpos(j)[:3]) for name, j in movable}
    for step in range(30):
        for _, joint_name in movable:
            env.sim.data.set_joint_qvel(joint_name, [velocity, 0.0, 0.0, 0.0, 0.0, 0.0])
        env.sim.forward()
        env.step(dummy_action)
    end = {name: np.array(env.sim.data.get_joint_qpos(j)[:3]) for name, j in movable}

    disp = {name: float(np.linalg.norm(end[name] - start[name])) for name in start}
    print(f"v={velocity:.2f} m/s -> displacement after 30 steps: {disp}")
    env.close()
