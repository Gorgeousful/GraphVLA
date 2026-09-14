import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from examples.libero.eval import client as evaluation
from script.server import InferenceServer, InferenceSession, TopLevelTaskPlanner
from src.common.observation_wire import decode_observation, encode_observation


def test_point_policy_oracle_switches_atomic_inputs_after_release(tmp_path):
    from script.server import PointPolicyInferenceServer

    server = object.__new__(PointPolicyInferenceServer)
    server.sessions = {}
    server.planner = TopLevelTaskPlanner(dataset_dir=tmp_path)
    server._validate_request = lambda request: None
    server.execute_chunk_len = 10
    server.embodiment = SimpleNamespace(release_actions=lambda *args: [[-1]])
    calls = []
    server._predict_action = lambda request, **kwargs: calls.append(request) or {"action": [[1]]}
    request = {"session_id": "pp", "benchmark": "libero", "language": "sequence",
               "switch_mode": "oracle"}
    session = server._session_for(request)
    session.taskstructure = {"subtasks": [{"subtask": "first"}, {"subtask": "second"}]}
    assert server.infer_from_observation(request)["subtask_index"] == 0
    assert calls[-1]["language"] == "first"
    assert calls[-1]["session_id"] == "pp::atomic"
    release = server.infer_from_observation({**request, "completed_subtask_index": 0})
    assert release["release_pending"] and len(calls) == 1
    response = server.infer_from_observation({**request, "completed_subtask_index": 0})
    assert response["subtask_switched"] and response["subtask_index"] == 1
    assert calls[-1]["language"] == "second" and calls[-1]["reset"]
    server.infer_from_observation({**request, "completed_subtask_index": 1})
    assert server.infer_from_observation(request)["episode_done"]
    assert len(calls) == 2
    with pytest.raises(ValueError, match="oracle"):
        server.infer_from_observation({**request, "switch_mode": "predicted"})


def test_point_policy_reuses_sequence_structure_without_atomic_cache(tmp_path):
    from script.server import PointPolicyInferenceServer

    server = object.__new__(PointPolicyInferenceServer)
    server.sessions = {}
    server.planner = TopLevelTaskPlanner(dataset_dir=tmp_path)
    subtasks = [{"subtask": name, "nodes": [{"name": name, "role": "patient"}]}
                for name in ("put the small blue box right", "put the bottle on cabinet")]
    server.planner.task_cache["sequence"] = {"task": "sequence", "subtasks": subtasks}
    server._validate_request = lambda request: None
    server.execute_chunk_len = 10
    server.embodiment = SimpleNamespace(release_actions=lambda *args: [[-1]],
                                       to_action=lambda *args: [[1]])
    seen = []

    def build(request, session, structure):
        seen.append(session.taskstructure)
        session.feature_history.append({})
        return {}

    server.preprocessor = SimpleNamespace(build=build)
    server._infer_model = lambda *args, **kwargs: ({}, None)
    server.inference = SimpleNamespace(to_json=lambda value: value)
    server._tracking_response = lambda *args: (None, None, None)
    request = {"session_id": "pp", "benchmark": "libero", "language": "sequence",
               "switch_mode": "oracle", "reset": True}
    server.infer_from_observation(request)
    request.pop("reset")
    server.infer_from_observation(request)
    server.infer_from_observation({**request, "completed_subtask_index": 0})
    server.infer_from_observation(request)
    assert [entry["subtasks"] for entry in seen] == [[subtasks[0]], [subtasks[0]], [subtasks[1]]]
    assert set(server.planner.task_cache) == {"sequence"}


def test_ordered_progress_never_backfills_early_goals_or_loses_history():
    progress = evaluation.SequenceProgress([False] * 3)
    progress.update([False, True, False], 1)
    progress.update([True, True, False], 2)
    progress.update([True, True, True], 3)
    assert progress.completion_steps == [2]
    assert progress.progress == pytest.approx(1 / 3)
    assert progress.success
    progress.update([True, False, False], 4)
    progress.update([True, True, False], 5)
    progress.update([True, True, True], 6)
    assert progress.completion_steps == [2, 5, 6]
    assert progress.success
    progress.update([False, True, True], 7)
    assert progress.progress == 1
    assert not progress.success


@pytest.mark.parametrize("mode", ["oracle", "predicted"])
def test_planner_uses_only_selected_signal_and_advances_after_release(tmp_path, mode):
    planner = TopLevelTaskPlanner(dataset_dir=tmp_path, progress_window=2)
    session = InferenceSession("test", "libero", "sequence", switch_mode=mode)
    session.taskstructure = {"subtasks": [{"subtask": str(i)} for i in range(3)]}
    if mode == "oracle":
        for _ in range(3):
            assert not planner.update_after_inference({"subtask_progress": 1.0}, session)
        assert planner.update_after_inference({"subtask_progress": 0.0}, session, 0)
        assert not planner.update_after_inference({}, session, 0)
    else:
        with pytest.raises(ValueError, match="oracle"):
            planner.update_after_inference({}, session, 0)
        assert not planner.update_after_inference({"subtask_progress": 1.0}, session)
        assert planner.update_after_inference({"subtask_progress": 1.0}, session)
    assert session.subtask_index == 0 and session.release_pending
    session.feature_history.append({"old": True})
    assert planner.advance_after_release(session)
    assert session.subtask_index == 1 and not session.feature_history
    if mode == "oracle":
        assert not planner.update_after_inference({}, session, 0)
        with pytest.raises(ValueError, match="ahead"):
            planner.update_after_inference({}, session, 2)
        for index in (1, 2):
            assert planner.update_after_inference({}, session, index)
            planner.advance_after_release(session)
        assert session.task_complete


def test_session_mode_is_fixed_until_reset():
    server = object.__new__(InferenceServer)
    server.sessions = {}
    request = {"session_id": "test", "benchmark": "libero", "language": "sequence", "switch_mode": "oracle"}
    session = server._session_for(request)
    with pytest.raises(ValueError, match="cannot change"):
        server._session_for({**request, "switch_mode": "predicted"})
    server._session_for({**request, "switch_mode": "predicted", "reset": True})
    assert session.switch_mode == "predicted"


@pytest.mark.parametrize("mode", ["oracle", "predicted"])
def test_client_wire_signal_and_release_chunk_are_isolated(monkeypatch, mode):
    client = evaluation.InferenceClient(host="unused", port=0, switch_mode=mode)
    client.intrinsic, client.extrinsic = np.eye(3), np.eye(4)
    observation = {"agentview_image": np.zeros((2, 2, 3)), "wrist_image": np.zeros((2, 2, 3)),
                   "agentview_metric_depth": np.ones((2, 2)), "state": np.zeros(8)}
    requests = []

    async def reply(uri, payload):
        request = decode_observation(encode_observation(payload))
        requests.append(request)
        return {"action": [[0.] * 7] * 3, "switch_mode": mode, "subtask_index": 0,
                "release_pending": "completed_subtask_index" in request, "episode_done": False}

    monkeypatch.setattr(client, "_websocket_json", reply)
    client.infer(observation, "sequence")
    assert len(client.action_chunk) == 2
    client.notify_subtask_complete(0)
    if mode == "oracle":
        assert not client.action_chunk
        client.infer(observation, "sequence")
        assert requests[-1]["completed_subtask_index"] == 0
        client.notify_subtask_complete(0)
        assert len(client.action_chunk) == 2  # Do not truncate release-and-lift.
    else:
        assert len(client.action_chunk) == 2
    client.infer(observation, "sequence")
    client.infer(observation, "sequence")
    client.infer(observation, "sequence")
    assert "completed_subtask_index" not in requests[-1]
    client.reset_episode()
    assert client.completed_subtask_index is None


def test_sequence_cli_requires_explicit_positive_budget(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["client", "--task-suite-name", "arbitrary_suite"])
    with pytest.raises(ValueError, match="explicit --max-steps"):
        evaluation._resolve_max_steps(evaluation.parse_args(), has_sequences=True)
    monkeypatch.setattr(sys, "argv", ["client", "--task-suite-name", "arbitrary_suite",
                                     "--max-steps", "1500", "--switch-mode", "oracle"])
    args = evaluation.parse_args()
    assert args.max_steps == 1500 and args.switch_mode == "oracle"
    assert evaluation._resolve_max_steps(args, has_sequences=True) == 1500
    assert evaluation._resolve_max_steps(evaluation.Args(task_suite_name="libero_custom_0902"), has_sequences=False) == 520


def test_sequence_mapping_is_required_only_when_taskstructure_has_multiple_subtasks(monkeypatch, tmp_path):
    import libero.libero

    monkeypatch.setattr(libero.libero, "get_libero_path", lambda key: str(tmp_path))
    folder = tmp_path / "arbitrary_suite"
    folder.mkdir()
    task = SimpleNamespace(name="long_task", problem_folder=folder.name)
    assert evaluation._sequence_spec(task, 1) is None
    with pytest.raises(ValueError, match="Missing ordered environment goals"):
        evaluation._sequence_spec(task, 2)
    path = folder / "split.json"
    path.write_text(json.dumps({"train": ["long_task"]}))
    assert evaluation._sequence_spec(task, 1) is None
    with pytest.raises(ValueError, match="Missing ordered environment goals"):
        evaluation._sequence_spec(task, 2)
    spec = {"name": "long_task", "task_id": 99, "ordered_goals": ["(On a x)", "(On b y)"]}
    path.write_text(json.dumps({"sequences": [
        {"name": "single_task", "ordered_goals": ["(On a x)"]}, spec,
    ]}))
    assert evaluation._sequence_spec(task, 2) == spec
    assert evaluation._sequence_spec(task, 1) is None  # Metadata cannot override the taskstructure count.
    with pytest.raises(ValueError, match="count does not match"):
        evaluation._sequence_spec(task, 3)
    assert evaluation._sequence_spec(SimpleNamespace(name="single_task", problem_folder=folder.name), 1) is None
    assert evaluation._sequence_spec(SimpleNamespace(name="undeclared_task", problem_folder=folder.name), 1) is None
    path.write_text(json.dumps({"sequences": [spec, spec]}))
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation._sequence_spec(task, 2)


def test_task_info_uses_server_taskstructure_and_does_not_create_session(monkeypatch, tmp_path):
    server = object.__new__(InferenceServer)
    server.inference_lock = threading.Lock()
    server.sessions = {}
    server.planner = TopLevelTaskPlanner(dataset_dir=tmp_path)
    server.planner.task_cache = {
        "atomic": {"subtasks": [{"subtask": "atomic"}]},
        "sequence": {"subtasks": [{"subtask": "first"}, {"subtask": "second"}]},
    }

    async def reply(uri, payload):
        request = decode_observation(encode_observation(payload))
        assert request["type"] == "task_info"
        return server.task_info(request["languages"])

    monkeypatch.setattr(evaluation.InferenceClient, "_websocket_json", reply)
    args = evaluation.Args(task_suite_name="arbitrary_suite")
    evaluation._load_taskstructures(args, ["atomic", "sequence"])
    assert [len(args.taskstructures[name]["subtasks"]) for name in ("atomic", "sequence")] == [1, 2]
    assert not server.sessions


@pytest.mark.parametrize("mode", ["oracle", "predicted"])
def test_client_server_three_stage_release_handshake(monkeypatch, tmp_path, mode):
    server = object.__new__(InferenceServer)
    server.sessions = {}
    server.execute_chunk_len = 2
    server.planner = TopLevelTaskPlanner(dataset_dir=tmp_path, progress_window=1)
    server.planner.task_cache["sequence"] = {"subtasks": [{"subtask": str(i)} for i in range(3)]}
    inputs = {"entity_points": np.zeros(1), "entity_point_mask": np.zeros(1)}

    def build(request, session, subtask):
        session.feature_history.append({"frame": 0})
        return inputs

    server.preprocessor = SimpleNamespace(
        build=build, _append_feature_history=lambda session, frame: session.feature_history.append(frame),
        _feature_window=lambda session: session.feature_history, _build_model_input=lambda *args: inputs,
    )
    server.inference = SimpleNamespace(to_json=lambda value: value.tolist() if isinstance(value, np.ndarray) else value)
    server._infer_model = lambda *a, **kw: ({"subtask_progress": 1. if mode == "predicted" else 0.}, None)
    server._initial_points_response = lambda session: (None, None, None)
    server._tracking_response = lambda *args: (None, None, None)
    server.embodiment = SimpleNamespace(
        embodiment="franka_panda", to_action=lambda *a: [[0.] * 7] * 2,
        release_actions=lambda *a: [[0.] * 6 + [-1.]] * 4,
    )
    client = evaluation.InferenceClient(host="unused", port=0, switch_mode=mode)
    client.intrinsic, client.extrinsic = np.eye(3), np.eye(4)
    requests, releases = [], []

    async def reply(uri, payload):
        request = decode_observation(encode_observation(payload))
        requests.append(request)
        return server.infer_from_observation(request)

    monkeypatch.setattr(client, "_websocket_json", reply)
    observation = {"agentview_image": np.zeros((2, 2, 3)), "wrist_image": np.zeros((2, 2, 3)),
                   "agentview_metric_depth": np.ones((2, 2)), "state": np.zeros(8)}
    for _ in range(30):
        action = client.infer(observation, "sequence")
        if client.episode_done:
            break
        index = client.last_response["subtask_index"]
        if action[-1] == -1:
            releases.append(index)
        if mode == "oracle":
            client.notify_subtask_complete(index)
    assert client.episode_done
    assert releases == [0] * 4 + [1] * 4 + [2] * 4
    if mode == "predicted":
        assert all("completed_subtask_index" not in request for request in requests)
    else:
        assert [request["completed_subtask_index"] for request in requests
                if "completed_subtask_index" in request] == [0, 1, 2]


def test_resume_rejects_different_switch_mode_or_budget(tmp_path):
    protocol = evaluation._evaluation_protocol(evaluation.Args(task_suite_name="arbitrary_suite"), 1500, scoring="sequence_final_goals_v1")
    path = tmp_path / "episode.json"
    path.write_text(json.dumps({"metadata": {"task_id": 0, "episode_id": 0, "init_state_id": 0},
                                "result": {"evaluation_protocol": protocol}}))
    for change in ({"switch_mode": "oracle"}, {"max_steps": 1600}, {"scoring": "ordered_prefix_v1"}, {"scoring": "ordered_prefix_v2"}):
        with pytest.raises(ValueError, match="protocol mismatch"):
            evaluation._restore_episode_result(path, task_id=0, episode_idx=0, init_state_id=0,
                                               evaluation_protocol={**protocol, **change})


@pytest.mark.parametrize("statuses,expected_pr,expected_sr", [
    ([[True, False, False], [True, True, False], [True, True, True]], 1., True),
    ([[True, False, False], [True, True, False], [True, True, True], [False, True, True]], 1., False),
    ([[False, True, False], [True, True, False], [True, True, True]], 1 / 3, True),
    ([[True, False, False], [False, True, False], [False, False, True], [False, False, False]], 1., False),
    ([[False, False, False]], 0., False),
])
@pytest.mark.parametrize("finish", [True, False])
@pytest.mark.parametrize("declared_sequence", [True, False])
def test_rollout_scores_final_goals_with_ordered_pr_and_preserves_atomic_behavior(
    monkeypatch, tmp_path, statuses, expected_pr, expected_sr, finish, declared_sequence,
):
    import libero.libero

    goals = [["on", name, "target"] for name in ("a", "b", "c")]
    folder = tmp_path / "arbitrary_suite"
    folder.mkdir()
    (folder / "split.json").write_text(json.dumps({"sequences": [{
        "name": "sequence", "split": "Seq-ID", "ordered_goals": ["(" + " ".join(g) + ")" for g in goals],
    }] if declared_sequence else []}))
    monkeypatch.setattr(libero.libero, "get_libero_path", lambda key: str(tmp_path))
    observation = {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8)}

    class Environment:
        def __init__(self):
            self.env = self
            self.parsed_problem = {"goal_state": goals}
            self.index = 0
            self.status = [False] * 3

        def _eval_predicate(self, goal):
            return self.status[goals.index(goal)]

        def reset(self):
            pass

        def set_init_state(self, state):
            return observation

        def step(self, action):
            self.status = statuses[self.index]
            self.index += 1
            return observation, 0., all(self.status), {}

        def close(self):
            pass

    env = Environment()
    client = SimpleNamespace(worker_id=0, last_response=None, episode_done=False,
                             last_action_frame_id=0, intrinsic=np.eye(3),
                             reset_episode=lambda: None, set_camera=lambda **kw: None)

    def infer(*args):
        client.episode_done = finish and env.index == len(statuses)
        client.last_response = {"subtask_index": min(env.index, 2), "episode_done": client.episode_done}
        return np.zeros(7)

    client.infer = infer
    monkeypatch.setattr(evaluation, "_get_libero_env", lambda *a, **kw: (env, "sequence"))
    monkeypatch.setattr(evaluation, "_prepare_observation", lambda *a, **kw: {"state": np.zeros(8)})
    monkeypatch.setattr(evaluation, "_draw_response_points", lambda frame, *a, **kw: frame)
    monkeypatch.setattr(evaluation, "_record_object_poses", lambda env: {})
    monkeypatch.setattr(evaluation, "_to_libero_action", lambda action, **kw: action)
    suite = SimpleNamespace(get_task=lambda i: SimpleNamespace(name="sequence", language="sequence", problem_folder=folder.name),
                            get_task_init_states=lambda i: [0])
    result = evaluation._evaluate_task(
        args=evaluation.Args(task_suite_name=folder.name, num_steps_wait=0, num_trials_per_task=1, save_video=False,
                             taskstructures={"sequence": {"subtasks": [{}] * (3 if declared_sequence else 1)}}),
        task_suite=suite, task_order=1, total_tasks=1, task_id=0, max_steps=len(statuses) + int(finish),
        progress_threshold=.9, client=client, video_dir=tmp_path,
        overall_progress=SimpleNamespace(complete_episode=lambda: None), existing_records={},
    )["episodes"][0]
    if declared_sequence:
        assert client.switch_mode == "predicted"
        assert env.index == len(statuses)
        assert result["progress"] == pytest.approx(expected_pr)
        assert result["success"] == expected_sr
        assert result["final_goal_status"] == statuses[-1]
        assert result["end_reason"] == ("server_complete" if finish else "timeout")
        assert result["server_success"] == finish
    else:
        assert client.switch_mode is None
        first_success = next((i + 1 for i, status in enumerate(statuses) if all(status)), None)
        assert env.index == (first_success or len(statuses))
        assert result["progress"] == pytest.approx(sum(env.status) / len(env.status))
        assert result["success"] == (first_success is not None)
        assert result["end_reason"] == ("environment_success" if first_success else "server_complete" if finish else "timeout")
        assert "completed_prefix" not in result
