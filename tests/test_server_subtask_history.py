import numpy as np

from script.server import InferenceServer, InferenceSession, TopLevelTaskPlanner


def test_subtask_advances_only_after_release_and_discards_old_history(tmp_path):
    planner = TopLevelTaskPlanner(
        dataset_dir=tmp_path,
        complete_threshold=0.5,
        complete_window=1,
    )
    tracked_points = np.ones((2, 32, 3), dtype=np.float32)
    session = InferenceSession(
        session_id="episode-1",
        benchmark="libero",
        language="two-step task",
        taskstructure={"subtasks": [{"subtask": "first"}, {"subtask": "second"}]},
        feature_history=[{"frame_id": np.asarray([0])}],
        tracked_points=tracked_points,
    )

    release_requested = planner.update_after_inference(
        outputs={"is_complete": [[0.9]]},
        session=session,
    )

    assert release_requested is True
    assert session.release_pending is True
    assert session.subtask_index == 0
    assert len(session.feature_history) == 1

    switched = planner.advance_after_release(session)

    assert switched is True
    assert session.release_pending is False
    assert session.subtask_index == 1
    assert session.feature_history == []
    assert session.tracked_points is tracked_points


def test_completion_window_counts_consecutive_current_observations(tmp_path):
    planner = TopLevelTaskPlanner(
        dataset_dir=tmp_path,
        complete_threshold=0.5,
        complete_window=2,
    )
    session = InferenceSession(
        session_id="episode-1",
        benchmark="libero",
        language="one-step task",
        taskstructure={"subtasks": [{"subtask": "first"}]},
    )
    outputs = {"is_complete": [[[0.9]]]}

    frame_scores = planner._completion_frame_scores(outputs)
    first_release = planner.update_after_inference(
        outputs=outputs,
        session=session,
    )
    second_release = planner.update_after_inference(
        outputs=outputs,
        session=session,
    )

    assert frame_scores == [(0, 0.9)]
    assert first_release is False
    assert second_release is True
    assert session.release_pending is True
    assert session.task_complete is False

    assert planner.advance_after_release(session) is False
    assert session.release_pending is False
    assert session.task_complete is True


class _SwitchPreprocessor:
    def __init__(self) -> None:
        self.rebuild_frame_ids = None
        self.build_subtasks = []

    def build(self, request, session, subtaskstructure):
        self.build_subtasks.append(subtaskstructure["subtask"])
        session.feature_history.extend([
            {"frame_id": np.asarray([0])},
            {"frame_id": np.asarray([1])},
        ])
        return self._build_model_input(session, session.feature_history, subtaskstructure)

    def _append_feature_history(self, session, features):
        session.feature_history.append(features)

    def _feature_window(self, session):
        assert len(session.feature_history) == 1
        return [session.feature_history[0], session.feature_history[0]]

    def _build_model_input(self, session, frames, subtaskstructure):
        frame_ids = [int(frame["frame_id"][0]) for frame in frames]
        if session.subtask_index == 1:
            self.rebuild_frame_ids = frame_ids
        return {
            "entity_points": frame_ids,
            "entity_point_mask": frame_ids,
            "scene_condition_texts": [subtaskstructure["subtask"], None],
            "entity_role_condition_texts": ["actor", "patient", "target"],
        }


class _SwitchInference:
    def __init__(self) -> None:
        self.calls = []

    def infer(self, model_input, *, return_model_input):
        assert return_model_input is False
        self.calls.append(model_input)
        if len(self.calls) == 1:
            return {"is_complete": [[0.9]], "plan_id": "old"}
        return {"is_complete": [[0.0]], "plan_id": "new"}

    @staticmethod
    def to_json(value):
        return value


class _SwitchEmbodiment:
    future_horizon = 1

    def __init__(self) -> None:
        self.plan_ids = []

    def to_action(self, outputs, model_input, request, session):
        self.plan_ids.append(outputs["plan_id"])
        return [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]], [0.08]

    def release_actions(self, request, session, chunk_len):
        session.gripper_command = -1.0
        return [[9.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0] for _ in range(chunk_len)]


def test_server_releases_then_starts_next_subtask_from_repeated_release_end_frame(tmp_path):
    planner = TopLevelTaskPlanner(
        dataset_dir=tmp_path,
        complete_threshold=0.5,
        complete_window=1,
    )
    planner.task_cache["two-step task"] = {
        "subtasks": [{"subtask": "first"}, {"subtask": "second"}]
    }
    preprocessor = _SwitchPreprocessor()
    inference = _SwitchInference()
    embodiment = _SwitchEmbodiment()
    server = InferenceServer(
        host="localhost",
        port=0,
        execute_chunk_len=1,
        planner=planner,
        preprocessor=preprocessor,
        inference=inference,
        embodiment=embodiment,
    )
    request = {
        "benchmark": "libero",
        "session_id": "episode-1",
        "language": "two-step task",
        "observation.images.image": [],
        "observation.depth.metric": [],
        "observation.state": [],
        "camera.intrinsics": [],
        "camera.extrinsics": [],
    }

    release_response = server.infer_from_observation(request)

    assert len(inference.calls) == 1
    assert embodiment.plan_ids == []
    assert preprocessor.build_subtasks == ["first"]
    assert release_response["plan_id"] == "old"
    assert release_response["subtask"] == "first"
    assert release_response["subtask_index"] == 0
    assert release_response["subtask_switched"] is False
    assert release_response["episode_done"] is False
    assert release_response["action"][0][0] == 9.0

    next_response = server.infer_from_observation(request)

    assert len(inference.calls) == 2
    assert embodiment.plan_ids == ["new"]
    assert preprocessor.build_subtasks == ["first", "first"]
    assert preprocessor.rebuild_frame_ids == [1, 1]
    assert next_response["plan_id"] == "new"
    assert next_response["subtask"] == "second"
    assert next_response["subtask_index"] == 1
    assert next_response["subtask_switched"] is True
    assert next_response["action"][0][0] == 2.0


def test_final_subtask_releases_then_holds_without_more_model_inference(tmp_path):
    planner = TopLevelTaskPlanner(
        dataset_dir=tmp_path,
        complete_threshold=0.5,
        complete_window=1,
    )
    planner.task_cache["one-step task"] = {"subtasks": [{"subtask": "only"}]}
    preprocessor = _SwitchPreprocessor()
    inference = _SwitchInference()
    embodiment = _SwitchEmbodiment()
    server = InferenceServer(
        host="localhost",
        port=0,
        execute_chunk_len=1,
        planner=planner,
        preprocessor=preprocessor,
        inference=inference,
        embodiment=embodiment,
    )
    request = {
        "benchmark": "libero",
        "session_id": "episode-1",
        "language": "one-step task",
        "observation.images.image": [],
        "observation.depth.metric": [],
        "observation.state": [],
        "camera.intrinsics": [],
        "camera.extrinsics": [],
    }

    first_response = server.infer_from_observation(request)
    second_response = server.infer_from_observation(request)
    third_response = server.infer_from_observation(request)

    assert first_response["action"][0][0] == 9.0
    assert first_response["episode_done"] is False
    assert second_response["episode_done"] is True
    assert second_response["action"][0][0] == 9.0
    assert third_response["action"][0][0] == 9.0
    assert len(inference.calls) == 1
    assert server.sessions["episode-1"].task_complete is True


def test_tracking_response_prepends_current_actor_urdf_points() -> None:
    session = InferenceSession(
        session_id="episode-1",
        benchmark="libero",
        language="task",
    )
    current_features = {
        "gripper_points_xyz": np.asarray([
            [0.0, 0.0, 1.0],
            [0.2, 0.0, 2.0],
            [-0.2, 0.1, 2.0],
        ], dtype=np.float32),
        "intrinsic": np.asarray([
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 40.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32),
    }

    points, object_ids = InferenceServer._active_tracking_response(session, current_features)
    np.testing.assert_allclose(points, [[50.0, 40.0, 1.0], [60.0, 40.0, 1.0], [40.0, 45.0, 1.0]])
    np.testing.assert_array_equal(object_ids, [0, 0, 0])
