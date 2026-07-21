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
        model_input={"frame_query_frame_id": [[0]]},
        gripper_widths=[0.08],
        actions=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]],
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
    frame_ids = [[0]]
    outputs = {"is_complete": [[[0.9]]]}

    frame_scores = planner._completion_frame_scores(
        outputs,
        {"frame_query_frame_id": frame_ids},
    )
    first_switched = planner.update_after_inference(
        outputs=outputs,
        session=session,
        model_input={"frame_query_frame_id": frame_ids},
        gripper_widths=[0.08],
        actions=[[0.0] * 7],
    )
    second_switched = planner.update_after_inference(
        outputs=outputs,
        session=session,
        model_input={"frame_query_frame_id": frame_ids},
        gripper_widths=[0.08],
        actions=[[0.0] * 7],
    )

    assert frame_scores == [(0, 0.9)]
    assert first_switched is False
    assert second_switched is False


def test_completion_log_describes_the_returned_action_chunk(tmp_path, capsys):
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

    planner.update_after_inference(
        outputs={"is_complete": [[0.25]]},
        session=session,
        model_input={"frame_query_frame_id": [[0]]},
        gripper_widths=[0.08, 0.04, 0.0],
        actions=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )

    output = capsys.readouterr().out
    assert "query_frame=current complete_score=0.2500" in output
    assert "chunk_len=3" in output
    assert "gripper_widths=[0.0800, 0.0400, 0.0000]" in output
    assert "gripper_actions=[-1.0000, 0.0000, 1.0000]" in output
