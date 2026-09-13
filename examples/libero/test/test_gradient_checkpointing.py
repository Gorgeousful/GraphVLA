"""Checkpointing must preserve gradients, RNG state and BatchNorm updates."""

import random
from copy import deepcopy

import pytest
import torch
from torch import nn


@pytest.mark.parametrize("policy_name", ["act", "dp", "dp3", "graphpoint"])
def test_checkpointing_matches_training_and_saves_activations(policy_name):
    torch.set_num_threads(1)
    if policy_name == "act":
        from examples.libero.test.test_act_policy import make_policy
        model = make_policy()
        batch = {"images": torch.rand(2, 1, 3, 64, 64), "state": torch.randn(2, 8),
                 "actions": torch.randn(2, 3, 7), "is_pad": torch.zeros(2, 3, dtype=torch.bool),
                 "language_embedding": torch.randn(2, 8)}
    elif policy_name == "dp":
        from examples.libero.test.test_dp_policy import make_policy
        model = make_policy()
        batch = {"images": torch.rand(2, 2, 2, 3, 32, 32), "state": torch.randn(2, 2, 8),
                 "actions": torch.randn(2, 8, 7), "is_pad": torch.zeros(2, 8, dtype=torch.bool),
                 "language_embedding": torch.randn(2, 8)}
    elif policy_name == "dp3":
        from examples.libero.test.test_dp3_policy import make_policy
        model = make_policy()
        batch = {"point_cloud": torch.randn(2, 2, 8, 3), "state": torch.randn(2, 2, 8),
                 "actions": torch.randn(2, 4, 7), "is_pad": torch.zeros(2, 4, dtype=torch.bool),
                 "language_embedding": torch.randn(2, 8)}
    else:
        from examples.libero.test.test_progress_head import _model, _batch
        model, batch = _model(), _batch()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.2
        if isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.2
    initial = deepcopy(model.state_dict())
    runs = []
    for enabled in (False, True):
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)
        model.train()
        model.set_gradient_checkpointing(enabled)
        torch.manual_seed(123)
        random.seed(123)
        saved_bytes = 0
        def pack(tensor):
            nonlocal saved_bytes
            saved_bytes += tensor.numel() * tensor.element_size()
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss, _ = model(batch)
        loss.backward()
        runs.append((loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None},
                     {n: b.clone() for n, b in model.named_buffers()}, torch.get_rng_state(), random.getstate(), saved_bytes))
    plain, checked = runs
    torch.testing.assert_close(plain[0], checked[0])
    assert plain[1].keys() == checked[1].keys()
    for name in plain[1]:
        torch.testing.assert_close(plain[1][name], checked[1][name], msg=lambda msg: f"{name}: {msg}")
    for name in plain[2]:
        torch.testing.assert_close(plain[2][name], checked[2][name], rtol=0, atol=0)
    assert torch.equal(plain[3], checked[3]) and plain[4] == checked[4]
    assert checked[5] < plain[5]
    model.set_gradient_checkpointing(False)
    assert all(not m.gradient_checkpointing for m in model.modules())


def test_checkpointing_bypasses_eval_and_no_grad():
    from src.policy.checkpointing import checkpoint_module
    module = nn.Linear(4, 4)
    module.gradient_checkpointing = True
    calls = []
    module.register_forward_hook(lambda *args: calls.append(1))
    module.eval()
    checkpoint_module(module, torch.ones(2, 4)).sum().backward()
    assert len(calls) == 1
    module.train()
    with torch.no_grad():
        assert not checkpoint_module(module, torch.ones(2, 4)).requires_grad
    assert len(calls) == 2
