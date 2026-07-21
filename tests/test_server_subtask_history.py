import numpy as np

from script.server import InferenceServer, InferenceSession, TopLevelTaskPlanner


def test_advancing_subtask_discards_previous_subtask_history(tmp_path):
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

    switched = planner.update_after_inference(
        outputs={"is_complete": [[0.9]]},
        session=session,
    )

    assert switched is True
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
    first_switched = planner.update_after_inference(
        outputs=outputs,
        session=session,
    )
    second_switched = planner.update_after_inference(
        outputs=outputs,
        session=session,
    )

    assert frame_scores == [(0, 0.9)]
    assert first_switched is False
    assert second_switched is False


class _SwitchPreprocessor:
    def __init__(self) -> None:
        self.rebuild_frame_ids = None

    def build(self, request, session, subtaskstructure):
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


def test_server_discards_old_action_and_reinfers_from_current_frame(tmp_path):
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
        planner=planner,
        preprocessor=preprocessor,
        inference=inference,
        embodiment=embodiment,
    )
    request = {
        "benchmark": "libero",
        "session_id": "episode-1",
        "language": "two-step task",
        "execute_chunk_len": 1,
        "observation.images.image": [],
        "observation.state": [],
        "camera.intrinsics": [],
        "camera.extrinsics": [],
    }

    response = server.infer_from_observation(request)

    assert len(inference.calls) == 2
    assert embodiment.plan_ids == ["new"]
    assert preprocessor.rebuild_frame_ids == [1, 1]
    assert response["plan_id"] == "new"
    assert response["subtask"] == "second"
    assert response["subtask_index"] == 1
    assert response["subtask_switched"] is True
    assert response["action"][0][0] == 2.0
