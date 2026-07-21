from __future__ import annotations

from types import SimpleNamespace

import torch

from src.training import training


def test_build_model_compiles_with_static_shapes(monkeypatch) -> None:
    compile_kwargs = {}

    class FakeModel(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()

        def set_gradient_checkpointing(self, enabled: bool) -> None:
            self.gradient_checkpointing = enabled

    def fake_compile(model, **kwargs):
        compile_kwargs.update(kwargs)
        return model

    monkeypatch.setattr(training, "GraphFlowModel", FakeModel)
    monkeypatch.setattr(torch, "compile", fake_compile)
    model_config = SimpleNamespace(to_kwargs=lambda: {})
    training_config = SimpleNamespace(
        gradient_checkpointing=True,
        compile_model=True,
    )

    model = training.build_model(model_config, training_config, torch.device("cpu"))

    assert model.gradient_checkpointing is True
    assert compile_kwargs == {"dynamic": False}
