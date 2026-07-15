import numpy as np

from script.server import InferenceSession, TopLevelTaskPlanner


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
        model_input={"frame_query_frame_id": [[1]]},
        execute_chunk_len=1,
        gripper_widths=[0.08],
        actions=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]],
    )

    assert switched is True
    assert session.subtask_index == 1
    assert session.feature_history == []
    assert session.tracked_points is tracked_points
