from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.libero.config.graphpoint.model_config import ModelConfig
from script.server import EmbodimentAdapter
from src.policy.graphpoint.model import GraphFlowModel


@pytest.mark.parametrize("mode", ["points", "abs_action", "delta_action"])
def test_action_modes_train_and_sample_with_actor_history(mode):
    config = ModelConfig(
        action_mode=mode, actor_point_indices=(0, 1, 2, 5), num_points=6,
        cls_token_num=1, history_frames=(-1,), future_horizon=2,
        condition_dim=8, hidden_dim=32, encoder_layers=1, global_layer_types=(0,),
        flow_layers=1, num_heads=4, dropout=0.0,
    )
    assert config.action_delta == (mode == "delta_action")
    model = GraphFlowModel(**config.to_kwargs())
    dim = 13 if mode == "points" else 7
    batch = {
        "entity_points": torch.randn(2, 2, 3, 6, 3),
        "entity_point_mask": torch.ones(2, 2, 3, 6, dtype=torch.bool),
        "entity_presence": torch.ones(2, 3, dtype=torch.bool),
        "scene_condition": torch.randn(2, 2, 8),
        "gripper_closedness_history": torch.zeros(2, 2, 1),
        "target": {"trajectory": torch.randn(2, 2, dim), "subtask_progress": torch.ones(2, 1)},
    }
    loss, metrics = model(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.flow.output_projection.weight.grad.abs().sum() > 0
    if mode != "points":
        # AdaLN gates start at zero; history gets gradients after the first update.
        torch.optim.SGD(model.parameters(), lr=0.01).step()
        model.zero_grad(set_to_none=True)
        model(batch)[0].backward()
        assert model.flow.history_projection.weight.grad.abs().sum() > 0
        assert "loss_flow_action" in metrics
    else:
        assert not any("history_projection" in key for key in model.state_dict())
        legacy = GraphFlowModel(**{k: v for k, v in config.to_kwargs().items() if k != "action_mode"})
        legacy.load_state_dict(model.state_dict(), strict=True)
    outputs = model.eval().sample(batch, num_steps=1)
    key = "point_plan" if mode == "points" else "action_plan"
    assert outputs[key].shape == ((2, 2, 4, 3) if mode == "points" else (2, 2, 7))
    assert outputs["subtask_progress"].shape == (2, 1)
    assert torch.isfinite(outputs[key]).all()


@pytest.mark.parametrize("mode", ["abs_action", "delta_action"])
def test_action_adapter_bypasses_point_fitting_and_releases_in_correct_mode(mode):
    delta = mode == "delta_action"
    adapter = EmbodimentAdapter(
        future_horizon=2, actor_point_indices=(0, 1, 2, 5),
        action_mode=mode, action_delta=delta,
    )
    plan = np.array([[[0.2, -0.1, 1.2, 3.1, 0.0, 0.0, -0.4]] * 2])
    session = SimpleNamespace(benchmark="libero")
    actual = adapter.to_action({"action_plan": plan}, {}, session)
    np.testing.assert_allclose(actual, np.clip(plan[0], -1, 1) if delta else plan[0])
    release = np.asarray(adapter.release_actions(
        session, 2, {"observation.state": [0.2, -0.1, 1.2, 3.1, 0, 0, 0, 0]},
    ))
    np.testing.assert_allclose(release[:, -1], -1)
    np.testing.assert_allclose(release[:, 0], 0 if delta else 0.2)
    with pytest.raises(ValueError, match="finite action_plan"):
        adapter.to_action({"action_plan": np.full((1, 2, 7), np.nan)}, {}, session)


def test_action_mode_rejects_unknown_values():
    with pytest.raises(ValueError, match="action_mode"):
        ModelConfig(action_mode="pose")
