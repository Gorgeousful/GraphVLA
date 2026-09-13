import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples.libero.eval.client import _default_max_steps, _get_libero_env
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv


def test_suite_and_split():
    suite = benchmark.get_benchmark_dict()["libero_custom_0904"]()
    split = json.loads((Path(get_libero_path("bddl_files")) / suite.name / "split.json").read_text())
    names = suite.get_task_names()
    assert len(names) == 10 and names == split["task_order"]
    assert [names[i] for i in suite.train_task_ids] == [n for n in names if n in split["train"]]
    assert [names[i] for i in suite.ood_task_ids] == [n for n in names if n in split["ood"]]
    assert _default_max_steps(suite.name) == 520
    for i in range(10):
        assert len(suite.get_task_init_states(i)) == 50
    for a, b in ((0, 6), (1, 7), (2, 8), (3, 9), (3, 4), (3, 5)):
        np.testing.assert_array_equal(suite.get_task_init_states(a), suite.get_task_init_states(b))


@pytest.mark.parametrize("task_id", [4, 5])
def test_rotation_success_and_reference_reset(task_id):
    suite = benchmark.get_benchmark_dict()["libero_custom_0904"]()
    wrapper = ControlEnv(bddl_file_name=suite.get_task_bddl_file_path(task_id),
                         use_camera_obs=False, has_offscreen_renderer=False)
    try:
        wrapper.reset()
        wrapper.set_init_state(suite.get_task_init_states(task_id)[0])
        env = wrapper.env
        assert not wrapper.check_success()
        if task_id == 5:
            joint = env.fixtures_dict["flat_stove_1"].joints[0]
            for angle, expected in ((0.49, False), (0.51, True)):
                env.sim.data.set_joint_qpos(joint, angle)
                env.sim.forward()
                assert bool(wrapper.check_success()) == expected
        else:
            name = "cream_cheese_1"
            joint = env.objects_dict[name].joints[-1]
            pos = env.sim.data.get_joint_qpos(joint)[:3].copy()
            initial = env.rotation_reference[name].copy()
            for axis, angle, expected in (("z", 0.49, False), ("z", 0.51, True),
                                          ("z", -0.7, False), ("x", 0.8, False)):
                matrix = Rotation.from_euler(axis, angle).as_matrix() @ initial
                quat = np.roll(Rotation.from_matrix(matrix).as_quat(), 1)
                env.sim.data.set_joint_qpos(joint, np.r_[pos, quat])
                env.sim.forward()
                assert bool(wrapper.check_success()) == expected
            wrapper.set_init_state(env.sim.get_state().flatten())
            assert not wrapper.check_success()
            np.testing.assert_allclose(env.rotation_reference[name], matrix)
    finally:
        wrapper.close()


def test_graphvla_client_camera_observation():
    suite = benchmark.get_benchmark_dict()["libero_custom_0904"]()
    env, language = _get_libero_env(suite.get_task(4), 64, seed=0, control_freq=20)
    try:
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(4)[0])
        assert language == "rotate the cream cheese right"
        assert obs["agentview_image"].shape == (64, 64, 3)
        assert obs["agentview_depth"].shape[:2] == (64, 64)
        assert not env.check_success()
    finally:
        env.close()
