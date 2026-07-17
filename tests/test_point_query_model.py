from __future__ import annotations

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
