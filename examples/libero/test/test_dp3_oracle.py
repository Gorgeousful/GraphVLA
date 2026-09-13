import pytest

from script.server import DP3InferenceServer, TopLevelTaskPlanner


def test_oracle_uses_subtask_language_and_releases_before_advancing():
    server = object.__new__(DP3InferenceServer)
    server.sessions = {}
    server.execute_chunk_len = 10
    server.planner = object.__new__(TopLevelTaskPlanner)
    server.planner.task_cache = {"sequence": {"subtasks": [
        {"subtask": "first instruction"}, {"subtask": "second instruction"},
    ]}}
    languages = []
    server._predict_action = lambda r: (
        languages.append(r["language"]) or {"action": [[0.] * 7], "episode_done": False}
    )
    request = {"benchmark": "libero", "session_id": "one", "language": "sequence",
               "switch_mode": "oracle"}
    first = server.infer_from_observation(request)
    assert first["subtask_index"] == 0 and languages == ["first instruction"]
    release = server.infer_from_observation({**request, "completed_subtask_index": 0})
    assert release["release_pending"] and len(release["action"]) == 20
    assert all(action[-1] == -1 for action in release["action"])
    assert languages == ["first instruction"]
    second = server.infer_from_observation({**request, "completed_subtask_index": 0})
    assert second["subtask_index"] == 1 and second["subtask_switched"]
    assert languages[-1] == "second instruction"
    server.infer_from_observation({**request, "completed_subtask_index": 1})
    assert server.infer_from_observation(request)["episode_done"]
    with pytest.raises(ValueError):
        server.infer_from_observation({**request, "switch_mode": "predicted"})
    server.infer_from_observation({**request, "reset": True})
    assert languages[-1] == "first instruction"
    server.infer_from_observation({"language": "single task"})
    assert languages[-1] == "single task"
