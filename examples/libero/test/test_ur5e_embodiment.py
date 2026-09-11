from pathlib import Path
from types import SimpleNamespace
import threading

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from examples.libero.embodiment.robot import GeomRobot


@pytest.fixture(scope="module")
def geometry():
    return GeomRobot("ur5e", points_per_mesh=16)


def test_fixed_anchors_midpoint_and_pose_roundtrip(geometry):
    state = np.array([0.2, -0.1, 0.8, 0.3, -0.2, 0.5])
    camera = np.eye(4)
    camera[:3, :3] = R.from_rotvec([0.1, 0.2, -0.3]).as_matrix()
    camera[:3, 3] = [-0.2, 0.1, -0.4]
    anchors = []
    depths = []
    for width in (0.0, 0.03, 0.085):
        points = geometry.project_gripper_to_xyz(state, camera, width)
        local = geometry._gripper_keypoints_local(width)
        anchors.append(local[:3])
        depths.append(local[-1, 2])
        np.testing.assert_allclose(points[-1], (points[3] + points[4]) / 2, atol=1e-12)
        np.testing.assert_allclose(camera[:3, :3] @ points[-1] + camera[:3, 3], state[:3], atol=1e-12)
        assert local[1, 1] > local[2, 1]  # Semantic left is +Y, matching Panda.
        for indices in ((0, 1, 2), (0, 1, 2, 5), tuple(range(6))):
            recovered = geometry.project_actor_xyz_to_gripper(
                points[list(indices)], width, camera, actor_point_indices=indices)
            np.testing.assert_allclose(recovered[:6], state, atol=1e-9)
    np.testing.assert_allclose(anchors, np.broadcast_to(anchors[0], (3, 3, 3)), atol=1e-12)
    assert np.ptp(depths) > 0.01  # Midpoint offset must not use Panda's constant.


def test_actual_asymmetric_joint_state_is_used(geometry):
    joints = np.array([0.3, -0.28, 0.3, 0.35, -0.33, 0.34])
    state = np.r_[np.array([0.1, 0.2, 0.9, -0.1, 0.2, 0.4]), joints]
    points = geometry.project_observation_to_xyz(state, np.eye(4))
    recovered = geometry.project_actor_xyz_to_gripper(
        points, geometry.observation_gripper_width(state), actor_point_indices=tuple(range(6)),
        gripper_qpos=joints)
    np.testing.assert_allclose(recovered[:6], state[:6], atol=1e-9)
    with pytest.raises(ValueError, match="12|TCP pose"):
        geometry.project_observation_to_xyz(np.zeros(8), np.eye(4))


def test_robotiq_physics_opens_and_closes_stably():
    path = Path(__file__).resolve().parents[1] / "embodiment/ur5e/gripper.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    gaps = []
    for target in (0.0, 0.4, 0.8):
        mujoco.mj_resetData(model, data)
        data.ctrl[:] = target
        for _ in range(3000):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        assert np.linalg.norm(data.qvel) < 1e-5
        np.testing.assert_allclose(data.qpos, [target, -target, target] * 2, atol=0.003)
        tips = []
        for side in ("left", "right"):
            gid = model.geom(f"{side}_fingerpad_collision").id
            tips.append(data.geom_xpos[gid] + data.geom_xmat[gid].reshape(3, 3) @
                        [0, -model.geom_size[gid, 1], 0])
        gaps.append(np.linalg.norm(tips[0] - tips[1]))
    assert 0.084 < gaps[0] < 0.087
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[-1] < 0.002


def test_server_ur5e_geometry_and_action_recovery(geometry):
    from script.server import EmbodimentAdapter, InputPreprocessor, InferenceSession
    preprocessor = InputPreprocessor.__new__(InputPreprocessor)
    preprocessor.embodiment = "ur5e"
    preprocessor._robot_local = threading.local()
    preprocessor._robot_local.robot = geometry
    state = np.r_[0.1, -0.2, 0.8, 0.2, 0.1, -0.4, [0.2, -0.2, 0.2] * 2]
    points = preprocessor._state_to_gripper_points_xyz(SimpleNamespace(state=state, extrinsic=np.eye(4)))
    preprocessor.num_points = preprocessor.actor_num_points = 6
    preprocessor.actor_point_indices = tuple(range(6))
    preprocessor.point_coordinate_frame = "tcp_relative"
    preprocessor.norm_stats = {"tcp_relative_xyz": {"q01": [-1.] * 3, "q99": [1.] * 3}}
    frame = dict(tracks=np.zeros((0, 6, 3)), metric_depth=np.ones((2, 2)),
                 gripper_points_xyz=points, intrinsic=np.eye(3))
    subtask = dict(nodes=[], action_type="pick", action_degree=None)
    session = SimpleNamespace(taskstructure={"subtasks": [subtask]}, subtask_index=0)
    model_input = preprocessor._build_model_input(session, [frame], subtask)
    expected = 1 - 2 * geometry.observation_gripper_width(state) / geometry._MAX_GRIPPER_WIDTH
    np.testing.assert_allclose(model_input["gripper_closedness_history"], expected, atol=1e-6)
    np.testing.assert_allclose(model_input["tcp_origin"], state[None, :3], atol=1e-10)
    adapter = EmbodimentAdapter(future_horizon=1, actor_point_indices=tuple(range(6)),
                                robot_cls=GeomRobot, embodiment="ur5e")
    adapter._robot_local.robot = geometry
    outputs = {"point_plan": points[None, None], "point_plan_mask": np.ones((1, 1, 6), bool),
               "gripper_plan": np.array([[0.6]])}
    actions = adapter.to_action(outputs, {"observation.state": state, "camera.extrinsics": np.eye(4)},
                                InferenceSession(session_id="ur5e", benchmark="libero", language="pick"))
    np.testing.assert_allclose(actions[0][:6], state[:6], atol=1e-6)
    np.testing.assert_allclose(actions[0][6], 0.6, atol=1e-6)


def test_cli_and_server_embodiment_mismatch(monkeypatch):
    import sys
    from examples.libero.eval.client import parse_args, _server_info, InferenceClient
    monkeypatch.setattr(sys, "argv", ["client", "--embodiment", "ur5e"])
    assert parse_args().embodiment == "ur5e"
    monkeypatch.setattr(sys, "argv", ["client", "--embodiment", "ur5e", "--delta-action"])
    with pytest.raises(SystemExit):
        parse_args()

    async def info(uri, request):
        return {"ckpt_path": "/checkpoints/step_1.pt", "progress_threshold": 0.9, "embodiment": "franka_panda"}

    monkeypatch.setattr(InferenceClient, "_websocket_json", staticmethod(info))
    with pytest.raises(ValueError, match="embodiment mismatch"):
        _server_info(host="localhost", port=8001, embodiment="ur5e")


def test_libero_scene_transfer_and_live_controller_roundtrip(geometry):
    from examples.libero.eval.client import _get_libero_env, _prepare_observation, _to_libero_action, _dummy_action
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_custom_0902"]()
    env, _ = _get_libero_env(suite.get_task(0), 64, 42, 20, action_delta=False, embodiment="ur5e")
    try:
        env.reset()
        reset_qpos = env.sim.data.qpos.copy()
        initial = np.asarray(suite.get_task_init_states(0)[0])
        obs = env.set_init_state(initial)
        nq, nv = env._source_shape
        for data, src_data, (src, dst) in zip((env.sim.data.qpos, env.sim.data.qvel),
                                            (initial[1:1+nq], initial[1+nq:]), env._scene_indices):
            np.testing.assert_allclose(data[dst], src_data[src], atol=1e-12)
        robot_indices = env.robots[0]._ref_joint_pos_indexes
        np.testing.assert_allclose(env.sim.data.qpos[robot_indices], reset_qpos[robot_indices], atol=1e-12)
        for _ in range(5):
            action = _dummy_action(obs, action_delta=False, embodiment="ur5e", env=env)
            converted = _to_libero_action(action, action_delta=False, embodiment="ur5e", env=env)
            site = env.robots[0].eef_site_id
            np.testing.assert_allclose(converted[:3], env.sim.data.site_xpos[site], atol=1e-6)
            np.testing.assert_allclose(R.from_rotvec(converted[3:6]).as_matrix(),
                                       env.sim.data.site_xmat[site].reshape(3, 3), atol=1e-6)
            obs, _, _, _ = env.step(converted)
        prepared = _prepare_observation(obs, env, embodiment="ur5e")
        state = prepared["state"]
        points = geometry.project_observation_to_xyz(state, np.eye(4))
        prefix = env.robots[0].gripper.naming_prefix
        for index, name in ((0, "actor_frame"), (1, "left_outer_knuckle"), (2, "right_outer_knuckle")):
            np.testing.assert_allclose(points[index], env.sim.data.get_body_xpos(prefix + name), atol=1e-8)
        for index, side in ((3, "left"), (4, "right")):
            name = prefix + side + "_fingerpad_collision"
            gid = env.sim.model.geom_name2id(name)
            expected = env.sim.data.get_geom_xpos(name) + env.sim.data.get_geom_xmat(name) @ [0, -env.sim.model.geom_size[gid, 1], 0]
            np.testing.assert_allclose(points[index], expected, atol=1e-8)
        assert prepared["agentview_image"].shape == (64, 64, 3)
        assert state.shape == (12,)
    finally:
        env.close()
