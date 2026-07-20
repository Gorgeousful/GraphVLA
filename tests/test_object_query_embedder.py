import pytest
import torch

from src.model.embedding import LearnableFrameObjectPointEmbedding
from src.model.embedding import ObjectQueryEmbedder


def test_object_query_embedder_combines_shared_object_and_frame_embeddings() -> None:
    position_embedding = LearnableFrameObjectPointEmbedding(
        hidden_dim=4,
        max_objects=3,
        max_points=2,
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
        max_points=2,
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
        max_points=2,
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
