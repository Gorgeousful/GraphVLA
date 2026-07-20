from __future__ import annotations

import pytest
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
        num_object_query_types=2,
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
        query_type=torch.zeros_like(frame_id),
        head_names=("gripper_openness", "gripper_action"),
    )

    assert outputs["gripper_openness"].shape == (2, 3, 1)
    assert outputs["gripper_action"].shape == (2, 3, 1)


def test_infer_forwards_actor_query_type(monkeypatch) -> None:
    model = PointQueryModel(
        point_dim=3,
        num_points=2,
        actor_num_points=2,
        set_hidden_dim=4,
        set_layers=1,
        set_heads=2,
        condition_dim=None,
        encoder_hidden_dim=4,
        encoder_layers=1,
        encoder_heads=2,
        decoder_hidden_dim=4,
        decoder_layers=1,
        decoder_heads=2,
        output_dims={"gripper_action": 1},
        num_object_query_types=2,
        min_frame=-1,
        max_frame=1,
    )
    captured = {}
    monkeypatch.setattr(model, "encode", lambda **_: torch.zeros(1, 1, 4))
    monkeypatch.setattr(model, "decode", lambda **_: {})

    def capture_decode_object(**kwargs):
        captured.update(kwargs)
        return {"gripper_action": torch.zeros(1, 1, 1)}

    monkeypatch.setattr(model, "decode_object", capture_decode_object)
    query_type = torch.ones(1, 1, dtype=torch.long)
    model.infer(
        point_feats=torch.zeros(1, 1, 1, 2, 3),
        actor_feats=torch.zeros(1, 1, 1, 2, 3),
        object_id=torch.zeros(1, 1, dtype=torch.long),
        point_id=torch.zeros(1, 1, dtype=torch.long),
        frame_id=torch.zeros(1, 1, dtype=torch.long),
        actor_query_frame_id=torch.ones(1, 1, dtype=torch.long),
        actor_query_type=query_type,
        actor_head_names="gripper_action",
    )
    assert captured["query_type"] is query_type


def test_model_uses_separate_actor_and_object_point_embeddings() -> None:
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

    position_embedding = model.encoder_position_embedding
    assert position_embedding.actor_point_embed.weight.shape == (3, 16)
    assert position_embedding.object_point_embed.weight.shape == (4, 16)
    assert (
        position_embedding.actor_point_embed.weight.data_ptr()
        != position_embedding.object_point_embed.weight.data_ptr()
    )
    assert model.query_position_embedding is position_embedding
    assert model.query_embedder.position_embedding is position_embedding


def test_model_requires_shared_encoder_and_decoder_embedding_dimension() -> None:
    with pytest.raises(ValueError, match="must match"):
        PointQueryModel(
            encoder_hidden_dim=8,
            decoder_hidden_dim=16,
            output_dims={"point": 3},
        )
