from copy import deepcopy

import pytest
import torch

from examples.libero.config.graphpoint_gc.model_config import ModelConfig
from src.policy.registry import build_policy


@pytest.mark.parametrize("layers,output", [((0, 0), "current"), ((0, 1), "all")])
def test_collapse_symmetry_reference_isolation_and_training(layers, output):
    torch.manual_seed(7)
    config = ModelConfig(
        actor_point_indices=(0, 1, 2, 5), num_points=6, cls_token_num=6,
        history_frames=(-1,), future_horizon=2, condition_dim=8, hidden_dim=32,
        encoder_layers=2, global_layer_types=layers, encoder_output_type=output,
        flow_layers=1, num_heads=4, dropout=0.0,
    )
    model = build_policy(config).eval()
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
    batch["entity_point_mask"][1, :, 1, -1] = False
    memory, relation, semantic = model._encode(batch)
    assert memory.shape == (2, 6 if output == "current" else 12, 32)
    assert relation.shape == (2, 6, 32)
    assert model._memory_mask(batch) is None
    assert model._memory_positions(memory).numel() == memory.shape[1]

    changed_actor = deepcopy(batch)
    changed_actor["entity_points"][:, :, 0] += 5
    actor_memory, actor_relation, _ = model._encode(changed_actor)
    assert not torch.allclose(memory, actor_memory)
    torch.testing.assert_close(relation, actor_relation)
    grad_batch = deepcopy(batch)
    grad_batch["entity_points"].requires_grad_()
    _, selected_relation, _ = model._encode(grad_batch)
    model.progress_head(selected_relation.flatten(1), model._task_condition(semantic)).sum().backward()
    assert grad_batch["entity_points"].grad[:, :, 0].abs().sum() == 0
    assert grad_batch["entity_points"].grad[:, :, 1:].abs().sum() > 0
    model.zero_grad(set_to_none=True)
    all_config = deepcopy(config)
    all_config.progresshead_input = ["actor", "patient", "target"]
    all_model = build_policy(all_config).eval()
    all_model.load_state_dict(model.state_dict(), strict=True)
    assert not torch.allclose(all_model._encode(batch)[1], all_model._encode(changed_actor)[1])
    legacy_kwargs = config.to_kwargs()
    legacy_kwargs.pop("progresshead_input")
    legacy_model = type(model)(**legacy_kwargs).eval()
    legacy_model.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(legacy_model._encode(batch)[1], all_model._encode(batch)[1])

    # Swapping object roles cannot change a shared point-set representation.
    changed = deepcopy(batch)
    changed["entity_points"][1] = batch["entity_points"][1, :, [0, 2, 1]]
    changed["entity_point_mask"][1] = batch["entity_point_mask"][1, :, [0, 2, 1]]
    for expected, actual in zip((memory, relation, semantic), model._encode(changed)):
        torch.testing.assert_close(expected, actual, atol=2e-6, rtol=2e-5)

    changed = deepcopy(batch)
    changed["entity_points"][0, :, 2] = float("nan")
    changed["entity_points"][1, :, 1, -1] = float("nan")
    for expected, actual in zip((memory, relation, semantic), model._encode(changed)):
        torch.testing.assert_close(expected, actual)

    changed = deepcopy(batch)
    changed["initial_patient_points"] += 3
    altered_memory, altered_relation, _ = model._encode(changed)
    torch.testing.assert_close(memory, altered_memory)
    torch.testing.assert_close(relation[1], altered_relation[1])
    assert not torch.allclose(relation[0], altered_relation[0])
    noise = torch.randn(2, 2, 13)
    original = model.sample(batch, num_steps=1, noise=noise.clone())
    altered = model.sample(changed, num_steps=1, noise=noise.clone())
    torch.testing.assert_close(original["point_plan"], altered["point_plan"])
    torch.testing.assert_close(original["gripper_plan"], altered["gripper_plan"])
    assert not torch.allclose(original["subtask_progress"][0], altered["subtask_progress"][0])

    model.train()
    model.set_gradient_checkpointing(True)
    batch["initial_patient_points"].requires_grad_()
    loss, _ = model(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert batch["initial_patient_points"].grad[0].abs().sum() > 0
    assert batch["initial_patient_points"].grad[1].abs().sum() == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    clone = build_policy(config)
    clone.load_state_dict(model.state_dict(), strict=True)
    del batch["initial_patient_points"]
    with pytest.raises(ValueError, match="subtask-initial"):
        model._encode(batch)


def test_config_loading_keeps_gp_data_contract(tmp_path):
    from src.training.checkpoint import TrainingCheckpoint
    from src.training.training import load_example_configs

    data, model, training = load_example_configs("libero", "graphpoint_gc")
    gp_data, gp_model, gp_training = load_example_configs("libero", "graphpoint")
    assert data.episodes == gp_data.episodes
    assert data.tasks == gp_data.tasks
    assert data.horizon == gp_data.horizon
    assert data.action_delta == model.action_delta == False
    assert model.cls_token_num == 3 * gp_model.cls_token_num
    assert "node_attention_mode" not in model.to_kwargs()
    assert model.to_kwargs()["progresshead_input"] == ["patient", "target"]
    assert training.wandb_name != gp_training.wandb_name
    assert not training.resume
    checkpoint = TrainingCheckpoint(tmp_path, resume=False, keep_period=0)
    checkpoint.save_config_snapshots(data_config=data, model_config=model, training_config=training)
    loaded_data, loaded_model, _ = TrainingCheckpoint.load_config_snapshots(
        tmp_path / "checkpoints" / "step_1.pt",
    )
    assert loaded_model.policy_name == "graphpoint_gc"
    assert loaded_model.to_kwargs() == model.to_kwargs()
    assert loaded_data.episodes == data.episodes
