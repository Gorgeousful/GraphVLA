from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.policy.graphpoint.model import GraphFlowModel


@pytest.mark.parametrize("mode,layers,output", [
    ("full", (0, 0), "current"),
    ("role_chain", (0, 1), "all"),
    ("full", (1, 1), "all"),
    ("role_chain", (0, 0), "current"),
])
def test_binary_reference_only_changes_progress_and_absent_target_is_ignored(mode, layers, output):
    torch.manual_seed(7)
    model = GraphFlowModel(
        actor_point_indices=(0, 1, 2, 5), num_points=6, cls_token_num=2,
        history_horizon=1, future_horizon=2, condition_dim=8, hidden_dim=32,
        encoder_layers=2, global_layer_types=layers, encoder_output_type=output,
        node_attention_mode=mode, flow_layers=1, num_heads=4, dropout=0.0,
    ).eval()
    batch = {
        "entity_points": torch.randn(2, 2, 3, 6, 3),
        "entity_point_mask": torch.ones(2, 2, 3, 6, dtype=torch.bool),
        "entity_presence": torch.tensor([[True, True, False], [True, True, True]]),
        "scene_condition": torch.randn(2, 2, 8),
        "initial_patient_points": torch.randn(2, 6, 3),
        "initial_patient_mask": torch.ones(2, 6, dtype=torch.bool),
        "gripper_closedness_history": torch.zeros(2, 2, 1),
        "target": {"trajectory": torch.randn(2, 2, 13), "subtask_progress": torch.full((2, 1), 0.5)},
    }
    memory, relation, semantic = model._encode(batch)
    changed = deepcopy(batch)
    changed["initial_patient_points"] += 3.0
    changed_memory, changed_relation, _ = model._encode(changed)
    torch.testing.assert_close(memory, changed_memory)
    torch.testing.assert_close(relation[1], changed_relation[1])
    torch.testing.assert_close(relation[0, :2], changed_relation[0, :2])
    assert not torch.allclose(relation[0, 2:], changed_relation[0, 2:])

    changed = deepcopy(batch)
    changed["entity_points"][0, :, 2] = 1000.0
    altered = model._encode(changed)
    for expected, actual in zip((memory, relation, semantic), altered):
        torch.testing.assert_close(expected, actual)
    batch["initial_patient_points"].requires_grad_()
    model.train()
    model.set_gradient_checkpointing(mode == "role_chain")
    loss, _ = model(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert batch["initial_patient_points"].grad[0].abs().sum() > 0
    assert batch["initial_patient_points"].grad[1].abs().sum() == 0
    model.eval()
    noise = torch.randn(2, 2, 13)
    sampled = model.sample(batch, num_steps=1, noise=noise.clone())
    sampled_altered = model.sample(changed, num_steps=1, noise=noise.clone())
    torch.testing.assert_close(sampled["point_plan"], sampled_altered["point_plan"])
    assert torch.isfinite(sampled["subtask_progress"]).all()
    del batch["initial_patient_points"]
    with pytest.raises(ValueError, match="subtask-initial"):
        model._encode(batch)


def test_training_reference_uses_segment_start_before_task_filtering(monkeypatch):
    from datasets import Dataset
    from src.dataset import dataset as module

    class Source:
        hf_dataset = Dataset.from_dict({
            "episode_index": [0, 0, 0, 0, 1, 1],
            "subtask_id": [1, 1, 2, 2, 1, 1],
            "task_index": [0, 0, 0, 0, 1, 1],
            "node_points_xyz": np.arange(6 * 1 * 6 * 3).reshape(6, 1, 6, 3).tolist(),
            "valid_node_mask": [[True]] * 6,
            "subtask_node_mask": [[True]] * 6,
        })

        def __getitem__(self, index):
            return self.hf_dataset[index]

        def __len__(self):
            return len(self.hf_dataset)

    monkeypatch.setattr(module, "make_lerobot_dataset", lambda *args, **kwargs: Source())
    config = SimpleNamespace(dataset_dir="unused", episodes=None, video_backend="pyav",
                             tasks=[1], transforms=(), graphpoint_subtask_start=True)
    dataset = module.GenericDataset(config)
    assert dataset.subtask_start_indices.tolist() == [0, 0, 2, 2, 4, 4]
    sample = dataset[1]
    torch.testing.assert_close(sample["initial_node_points_xyz"],
                               torch.tensor(Source.hf_dataset[4]["node_points_xyz"]))


def test_online_reference_survives_history_eviction_and_resets_at_subtask_boundary():
    from script.server import InferenceSession, InputPreprocessor, TopLevelTaskPlanner

    preprocessor = object.__new__(InputPreprocessor)
    preprocessor.num_points = 6
    preprocessor.actor_point_indices = (0, 1, 2, 3, 4, 5)
    preprocessor.actor_num_points = 6
    preprocessor.history_frames = ()
    preprocessor.point_coordinate_frame = "tcp_relative"
    preprocessor.norm_stats = {"tcp_relative_xyz": {"q01": [-1.] * 3, "q99": [1.] * 3}}
    preprocessor._subtask_object_indices = lambda *args: [0]
    preprocessor._active_tracks = lambda tracks, indices: tracks
    preprocessor._object_feats = lambda tracks, *args: tracks
    subtask = {"action_type": "rotate", "action_degree": "right", "nodes": [{"role": "patient"}]}
    session = InferenceSession("test", "libero", "rotate")
    session.taskstructure = {"subtasks": [subtask, subtask]}

    def frame(value, tcp):
        tracks = np.zeros((2, 6, 3), dtype=np.float32)
        tracks[0] = value
        return {"tracks": tracks, "metric_depth": np.ones((2, 2)),
                "intrinsic": np.eye(3), "gripper_points_xyz": np.full((6, 3), tcp, dtype=np.float32)}

    preprocessor._append_feature_history(session, frame(0.2, 0.0))
    preprocessor._append_feature_history(session, frame(0.7, 0.1))
    model_input = preprocessor._build_model_input(session, preprocessor._feature_window(session), subtask)
    np.testing.assert_allclose(model_input["initial_patient_points"], 0.1, atol=1e-6)
    preprocessor._append_feature_history(session, frame(0.8, 0.15))
    model_input = preprocessor._build_model_input(session, preprocessor._feature_window(session), subtask)
    np.testing.assert_allclose(model_input["initial_patient_points"], 0.05, atol=1e-6)
    session.release_pending = True
    planner = object.__new__(TopLevelTaskPlanner)
    assert planner.advance_after_release(session)
    preprocessor._append_feature_history(session, frame(0.8, 0.15))
    model_input = preprocessor._build_model_input(session, preprocessor._feature_window(session), subtask)
    np.testing.assert_allclose(model_input["initial_patient_points"], 0.65, atol=1e-6)
    session.reset(benchmark="libero", language="rotate again")
    assert session.initial_patient_camera is None and session.subtask_initial_frame is None


def test_training_reference_uses_current_tcp_and_patient_role_not_first_node():
    from src.dataset.transform import CenterOnCurrentTCP, CustomTransform, Normalize

    stats = {"tcp_relative_xyz": {"q01": [-1.] * 3, "q99": [1.] * 3}}
    data = {
        "node_points_xyz": torch.full((4, 2, 6, 3), 0.7),
        "gripper_points_xyz": torch.full((4, 6, 3), 0.1),
        "valid_node_mask": torch.tensor([[False, True]] * 4),
        "subtask_node_mask": torch.tensor([[False, True]] * 4),
        "subtaskstructure": {"nodes": [{"role": "patient"}]},
        "initial_node_points_xyz": torch.full((2, 6, 3), 0.2),
        "initial_valid_node_mask": torch.tensor([False, True]),
        "initial_subtask_node_mask": torch.tensor([False, True]),
        "history_horizon": 1, "future_horizon": 2,
        "action": torch.zeros(4, 7), "subtask_progress": torch.zeros(4),
    }
    data = CenterOnCurrentTCP()(data)
    data = Normalize(stats, field_map={key: "tcp_relative_xyz" for key in (
        "node_points_xyz", "initial_node_points_xyz", "gripper_points_xyz",
    )})(data)
    transform = CustomTransform(mode="build_model_input", extra={
        "actor_point_indices": (0, 1, 2, 3, 4, 5), "norm_stats": stats,
        "point_stats_field": "tcp_relative_xyz",
    })
    transform._build_scene_condition = lambda *args, **kwargs: torch.zeros(2, 8)
    result = transform(data)
    torch.testing.assert_close(result["initial_patient_points"], torch.full((6, 3), 0.1))
    torch.testing.assert_close(result["entity_points"][:, 1], torch.full((2, 6, 3), 0.6))
    assert result["entity_presence"].tolist() == [True, True, False]
    assert result["initial_patient_mask"].all()


def test_occluded_target_does_not_become_binary():
    model = GraphFlowModel(
        actor_point_indices=(0, 1, 2, 5), num_points=6, history_horizon=0,
        future_horizon=2, condition_dim=8, hidden_dim=32, encoder_layers=1,
        flow_layers=1, num_heads=4, dropout=0.0,
    ).eval()
    batch = {"entity_points": torch.randn(1, 1, 3, 6, 3),
             "entity_point_mask": torch.ones(1, 1, 3, 6, dtype=torch.bool),
             "entity_presence": torch.ones(1, 3, dtype=torch.bool),
             "scene_condition": torch.randn(1, 2, 8)}
    batch["entity_point_mask"][:, :, 2] = False
    _, relation, _ = model._encode(batch)  # No initial reference is required.
    assert torch.isfinite(relation).all()
