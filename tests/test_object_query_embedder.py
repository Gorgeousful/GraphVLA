import pytest
import torch

from src.model.embedding import LearnableFrameObjectPointEmbedding
from src.model.embedding import ObjectQueryEmbedder


def test_object_query_embedder_combines_shared_object_and_frame_embeddings() -> None:
    position_embedding = LearnableFrameObjectPointEmbedding(
        hidden_dim=4,
        max_objects=3,
        max_actor_points=2,
        max_object_points=2,
        min_frame=-1,
        max_frame=1,
    )
    embedder = ObjectQueryEmbedder(hidden_dim=4, position_embedding=position_embedding)
    object_id = torch.tensor([[0, 1]])
    frame_id = torch.tensor([[0, 1]])

    output = embedder(object_id=object_id, frame_id=frame_id)
    expected = embedder.out_norm(
        position_embedding.object_embed(object_id)
        + position_embedding.frame_embed(frame_id - position_embedding.min_frame)
    )

    assert output.shape == (1, 2, 4)
    torch.testing.assert_close(output, expected)


def test_object_query_embedder_validates_query_shape() -> None:
    position_embedding = LearnableFrameObjectPointEmbedding(
        hidden_dim=4,
        max_objects=3,
        max_actor_points=2,
        max_object_points=2,
        min_frame=-1,
        max_frame=1,
    )
    embedder = ObjectQueryEmbedder(hidden_dim=4, position_embedding=position_embedding)

    with pytest.raises(ValueError, match=r"Expected object_id/frame_id \[B, Q\]"):
        embedder(object_id=torch.tensor([0]), frame_id=torch.tensor([0]))

    with pytest.raises(ValueError, match="must have the same shape"):
        embedder(object_id=torch.tensor([[0, 1]]), frame_id=torch.tensor([[0]]))


def test_object_query_embedder_adds_optional_query_type_embedding() -> None:
    position_embedding = LearnableFrameObjectPointEmbedding(
        hidden_dim=4,
        max_objects=3,
        max_actor_points=2,
        max_object_points=2,
        min_frame=-1,
        max_frame=1,
    )
    embedder = ObjectQueryEmbedder(
        hidden_dim=4,
        position_embedding=position_embedding,
        num_query_types=2,
    )
    object_id = torch.tensor([[0, 0]])
    frame_id = torch.tensor([[0, 0]])
    query_type = torch.tensor([[0, 1]])

    output = embedder(object_id=object_id, frame_id=frame_id, query_type=query_type)
    expected = embedder.out_norm(
        position_embedding.encode_object_query(object_id=object_id, frame_id=frame_id)
        + embedder.query_type_embed(query_type)
    )

    torch.testing.assert_close(output, expected)


def test_point_position_embedding_separates_actor_and_object_identities() -> None:
    position_embedding = LearnableFrameObjectPointEmbedding(
        hidden_dim=2,
        max_objects=3,
        max_actor_points=2,
        max_object_points=3,
        min_frame=0,
        max_frame=1,
    )
    with torch.no_grad():
        position_embedding.frame_embed.weight.zero_()
        position_embedding.object_embed.weight.zero_()
        position_embedding.actor_point_embed.weight.copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        )
        position_embedding.object_point_embed.weight.copy_(
            torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        )

    actor_grid = position_embedding.encode_grid(
        num_frames=1,
        num_objects=1,
        num_points=2,
        device=torch.device("cpu"),
        object_offset=0,
        point_type="actor",
    )
    object_grid = position_embedding.encode_grid(
        num_frames=1,
        num_objects=1,
        num_points=3,
        device=torch.device("cpu"),
        object_offset=1,
        point_type="object",
    )
    query = position_embedding.encode_query(
        object_id=torch.tensor([[0, 0]]),
        point_id=torch.tensor([[0, 1]]),
        frame_id=torch.ones(1, 2, dtype=torch.long),
    )

    torch.testing.assert_close(actor_grid[0, 0], position_embedding.actor_point_embed.weight)
    torch.testing.assert_close(object_grid[0, 0], position_embedding.object_point_embed.weight)
    torch.testing.assert_close(
        query[0],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )

    with pytest.raises(ValueError, match="only supports actor"):
        position_embedding.encode_query(
            object_id=torch.tensor([[1]]),
            point_id=torch.tensor([[0]]),
            frame_id=torch.tensor([[1]]),
        )
    with pytest.raises(ValueError, match="future frames"):
        position_embedding.encode_query(
            object_id=torch.tensor([[0]]),
            point_id=torch.tensor([[0]]),
            frame_id=torch.tensor([[0]]),
        )


def test_point_grid_rejects_inconsistent_point_type_and_object_ids() -> None:
    position_embedding = LearnableFrameObjectPointEmbedding(
        hidden_dim=2,
        max_objects=3,
        max_actor_points=2,
        max_object_points=3,
        min_frame=0,
        max_frame=0,
    )

    with pytest.raises(ValueError, match="object point grid"):
        position_embedding.encode_grid(
            num_frames=1,
            num_objects=1,
            num_points=2,
            device=torch.device("cpu"),
            object_offset=0,
            point_type="object",
        )
