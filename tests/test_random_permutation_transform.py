from __future__ import annotations

import torch

from src.dataset.transform import CustomTransform


def test_random_permutation_keeps_object_trajectories_and_targets_aligned() -> None:
    point_feats = torch.tensor(
        [
            [[[0.0], [1.0], [2.0], [3.0]], [[100.0], [101.0], [102.0], [103.0]]],
            [[[10.0], [11.0], [12.0], [13.0]], [[110.0], [111.0], [112.0], [113.0]]],
        ]
    )
    actor_feats = torch.arange(12, dtype=torch.float32).reshape(2, 1, 3, 2)

    object_id = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2] * 2)
    point_id = torch.tensor([0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3] * 2)
    frame_id = torch.tensor([-1] * 11 + [0] * 11)
    target_point = torch.tensor(
        [
            900, 901, 902, 20, 21, 22, 23, 120, 121, 122, 123,
            910, 911, 912, 30, 31, 32, 33, 130, 131, 132, 133,
        ],
        dtype=torch.float32,
    ).unsqueeze(-1)
    target_mask = torch.tensor(
        [
            True, True, True, True, False, True, False, False, True, False, True,
            True, True, True, False, True, False, True, True, False, True, False,
        ]
    )
    data = {
        "point_feats": point_feats,
        "actor_feats": actor_feats,
        "object_id": object_id,
        "point_id": point_id,
        "frame_id": frame_id,
        "target": {"point": target_point, "point_mask": target_mask},
    }

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        output = CustomTransform(mode="random_object_permutation")(data)

    permutation_1 = output["point_feats"][0, 0, :, 0].long()
    permutation_2 = (output["point_feats"][0, 1, :, 0] - 100).long()
    assert not (torch.equal(permutation_1, torch.arange(4)) and torch.equal(permutation_2, torch.arange(4)))
    torch.testing.assert_close(output["point_feats"][1, 0, :, 0], 10 + permutation_1.float())
    torch.testing.assert_close(output["point_feats"][1, 1, :, 0], 110 + permutation_2.float())

    object_1 = output["target"]["point"][object_id == 1].reshape(2, 4)
    object_2 = output["target"]["point"][object_id == 2].reshape(2, 4)
    expected_object_1 = torch.tensor([[20, 21, 22, 23], [30, 31, 32, 33]], dtype=torch.float32)
    expected_object_2 = torch.tensor([[120, 121, 122, 123], [130, 131, 132, 133]], dtype=torch.float32)
    torch.testing.assert_close(object_1, expected_object_1[:, permutation_1])
    torch.testing.assert_close(object_2, expected_object_2[:, permutation_2])

    expected_mask_1 = target_mask[object_id == 1].reshape(2, 4)[:, permutation_1]
    expected_mask_2 = target_mask[object_id == 2].reshape(2, 4)[:, permutation_2]
    torch.testing.assert_close(output["target"]["point_mask"][object_id == 1].reshape(2, 4), expected_mask_1)
    torch.testing.assert_close(output["target"]["point_mask"][object_id == 2].reshape(2, 4), expected_mask_2)

    torch.testing.assert_close(output["actor_feats"], actor_feats)
    torch.testing.assert_close(output["target"]["point"][object_id == 0], target_point[object_id == 0])
    for key in ("object_id", "point_id", "frame_id"):
        torch.testing.assert_close(output[key], data[key])
