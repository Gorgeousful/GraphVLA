from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from examples.libero.eval.client import _dummy_action, _get_libero_env, _to_libero_action
from script.server import EmbodimentAdapter


@pytest.mark.parametrize("embodiment", ["franka_panda", "ur5e", "sawyer"])
def test_delta_plan_runs_with_native_world_osc_goals(embodiment):
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_custom_0902"]()
    env, _ = _get_libero_env(suite.get_task(0), 64, 42, 20,
                             action_delta=True, embodiment=embodiment)
    try:
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(0)[0])
        controller = env.robots[0].controller
        assert controller.use_delta
        np.testing.assert_allclose(controller.output_max, [0.05] * 3 + [0.5] * 3)
        np.testing.assert_allclose(controller.output_min, [-0.05] * 3 + [-0.5] * 3)
        wait = _dummy_action(obs, action_delta=True, embodiment=embodiment, env=env)
        np.testing.assert_array_equal(wait, [0, 0, 0, 0, 0, 0, -1])
        adapter = EmbodimentAdapter(future_horizon=3, actor_point_indices=(0, 1, 2, 3, 4, 5),
                                   action_mode="delta_action", action_delta=True, embodiment=embodiment)
        plan = np.array([wait, [0.1, -0.2, 0.3, 0, 0, 0, 1],
                         [0, 0, 0, 0.1, -0.2, 0.3, -1]])
        actions = adapter.to_action({"action_plan": plan[None]}, {}, SimpleNamespace(benchmark="libero"))
        np.testing.assert_allclose(actions, plan)
        for action in actions:
            converted = _to_libero_action(action, action_delta=True, embodiment=embodiment, env=env)
            np.testing.assert_allclose(converted, action)
            controller.update(force=True)
            controller.reset_goal()
            controller.set_goal(converted[:6])
            np.testing.assert_allclose(controller.goal_pos, controller.ee_pos + converted[:3] * 0.05,
                                       atol=1e-7)
            np.testing.assert_allclose(controller.goal_ori,
                                       R.from_rotvec(converted[3:6] * 0.5).as_matrix() @ controller.ee_ori_mat,
                                       atol=1e-6)
            # Actual simulator execution exercises robot-specific arm and gripper control.
            obs, _, _, _ = env.step(converted)
            assert np.isfinite(env.sim.data.qpos).all()
            gripper = env.robots[0].gripper
            gripper.current_action = np.zeros_like(gripper.current_action)
            closing = gripper.format_action(np.array([1.0])).copy()
            gripper.current_action = np.zeros_like(gripper.current_action)
            opening = gripper.format_action(np.array([-1.0])).copy()
            np.testing.assert_allclose(closing, -opening)
            assert np.any(closing != 0)
    finally:
        env.close()
