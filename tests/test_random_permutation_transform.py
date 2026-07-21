from __future__ import annotations

import torch

from src.dataset.transform import CustomTransform


def test_random_permutation_preserves_object_tracks_and_actor_order() -> None:
    points = torch.zeros(2, 3, 4, 3)
    points[:, 0, :, 0] = torch.arange(4)
    points[0, 1, :, 0] = torch.arange(4)
    points[1, 1, :, 0] = 10 + torch.arange(4)
    points[0, 2, :, 0] = 100 + torch.arange(4)
    points[1, 2, :, 0] = 110 + torch.arange(4)
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    data = {"entity_points": points, "entity_point_mask": mask, "target": {"is_complete": torch.ones(1)}}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        output = CustomTransform(mode="random_object_permutation")(data)
    torch.testing.assert_close(output["entity_points"][:, 0], points[:, 0])
    patient_order = output["entity_points"][0, 1, :, 0].long()
    target_order = (output["entity_points"][0, 2, :, 0] - 100).long()
    torch.testing.assert_close(output["entity_points"][1, 1, :, 0], 10 + patient_order.float())
    torch.testing.assert_close(output["entity_points"][1, 2, :, 0], 110 + target_order.float())
    assert output["target"] is data["target"]
