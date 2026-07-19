from __future__ import annotations

import torch

from src.model.model import PointQueryModel


def test_actor_and_object_set_projections_have_independent_weights() -> None:
    model = PointQueryModel(
        point_dim=3,
        num_points=4,
        actor_num_points=3,
        set_hidden_dim=8,
        set_layers=1,
        set_heads=2,
        condition_dim=None,
        encoder_hidden_dim=16,
        encoder_layers=1,
        encoder_heads=2,
        decoder_hidden_dim=16,
        decoder_layers=1,
        decoder_heads=2,
        output_dims={"point": 3},
        min_frame=-1,
        max_frame=1,
    )

    state = model.state_dict()

    assert "object_set_proj.weight" in state
    assert "actor_set_proj.weight" in state
    assert "set_proj.weight" not in state
    assert (
        state["object_set_proj.weight"].data_ptr()
        != state["actor_set_proj.weight"].data_ptr()
    )


def test_object_query_decode_uses_shared_position_embedding() -> None:
    model = PointQueryModel(
        point_dim=3,
        num_points=4,
        actor_num_points=3,
        set_hidden_dim=8,
        set_layers=1,
        set_heads=2,
        condition_dim=None,
        encoder_hidden_dim=16,
        encoder_layers=1,
        encoder_heads=2,
        decoder_hidden_dim=16,
        decoder_layers=1,
        decoder_heads=2,
        output_dims={"point": 3, "gripper_openness": 1, "gripper_action": 1},
        min_frame=-1,
        max_frame=1,
    )

    assert model.query_embedder.position_embedding is model.query_position_embedding
    assert model.object_query_embedder.position_embedding is model.query_position_embedding
    assert model.frame_query_embedder.position_embedding is model.query_position_embedding
    assert not tuple(model.object_query_embedder.out_norm.parameters())

    memory = torch.randn(2, 5, 16)
    object_id = torch.zeros((2, 3), dtype=torch.long)
    frame_id = torch.tensor([[-1, 0, 1], [-1, 0, 1]])
    outputs = model.decode_object(
        memory=memory,
        object_id=object_id,
        frame_id=frame_id,
        head_names=("gripper_openness", "gripper_action"),
    )

    assert outputs["gripper_openness"].shape == (2, 3, 1)
    assert outputs["gripper_action"].shape == (2, 3, 1)
