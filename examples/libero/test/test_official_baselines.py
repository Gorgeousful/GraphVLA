"""Numerical comparisons against the optional local upstream checkouts."""

import importlib.util
from pathlib import Path

import pytest
import torch

from src.policy.dp.blocks import ConditionalUnet1d


RESEARCH = Path(__file__).resolve().parents[4]


def test_act_pretrained_frozen_bn_checkpoint_backward():
    from examples.libero.test.test_act_policy import make_policy
    from torchvision.ops.misc import FrozenBatchNorm2d
    model = make_policy()
    model.num_cameras = 2
    # Non-identity statistics reproduce the pretrained backbone's autograd path.
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.weight.fill_(0.7)
            module.running_mean.fill_(0.2)
    model.set_gradient_checkpointing(True)
    batch = dict(images=torch.rand(2, 2, 3, 64, 64), state=torch.rand(2, 8),
                 actions=torch.rand(2, 3, 7), is_pad=torch.zeros(2, 3, dtype=torch.bool),
                 language_embedding=torch.rand(2, 8))
    model(batch)[0].backward()
    assert torch.isfinite(model.backbones[0].features[0].weight.grad).all()


@pytest.mark.parametrize("policy", ["dp", "dp3"])
def test_unet_matches_upstream(policy, monkeypatch):
    checkout = RESEARCH / ("diffusion_policy" if policy == "dp" else
                           "3D-Diffusion-Policy/3D-Diffusion-Policy")
    package = "diffusion_policy" if policy == "dp" else "diffusion_policy_3d"
    if not checkout.exists():
        pytest.skip("Local upstream checkout unavailable")
    monkeypatch.syspath_prepend(str(checkout))
    module = __import__(package + ".model.diffusion.conditional_unet1d", fromlist=["ConditionalUnet1D"])
    upstream = module.ConditionalUnet1D(
        7, global_cond_dim=12, diffusion_step_embed_dim=16,
        down_dims=[16, 32, 64], kernel_size=5, n_groups=8,
        **({"cond_predict_scale": True} if policy == "dp" else {"condition_type": "film"}),
    )
    local = ConditionalUnet1d(7, 12, diffusion_step_embed_dim=16,
                              down_dims=(16, 32, 64), kernel_size=5, groups=8)
    mapped = {}
    for key, value in local.state_dict().items():
        name = key.replace("timestep_encoder", "diffusion_step_encoder").replace(
            "middle_modules", "mid_modules").replace("output.", "final_conv.")
        name = name.replace(".condition.", ".cond_encoder.").replace(".residual.", ".residual_conv.")
        parts = name.split(".")
        if parts[0] in ("down_modules", "up_modules") and parts[2] == "2":
            parts.insert(3, "conv")
        mapped[".".join(parts)] = value
    upstream.load_state_dict(mapped, strict=True)
    sample = torch.randn(2, 16, 7, requires_grad=True)
    condition = torch.randn(2, 12, requires_grad=True)
    timestep = torch.tensor([1, 3])
    actual = local(sample, timestep, condition)
    expected = upstream(sample, timestep, global_cond=condition)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.square().sum(), (sample, condition))
    expected_grads = torch.autograd.grad(expected.square().sum(), (sample, condition))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_act_transformer_matches_upstream_with_language_token():
    path = RESEARCH / "act/detr/models/transformer.py"
    if not path.exists():
        pytest.skip("Local upstream ACT checkout unavailable")
    spec = importlib.util.spec_from_file_location("upstream_act_transformer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from src.policy.act.transformer import ACTTransformer
    local = ACTTransformer(32, 4, 64, 2, 3, 0.0, False)
    upstream = module.Transformer(d_model=32, nhead=4, num_encoder_layers=2,
                                  num_decoder_layers=3, dim_feedforward=64,
                                  dropout=0.0, return_intermediate_dec=True)
    mapped = {}
    for key, value in local.state_dict().items():
        key = key.replace(".self_attention.", ".self_attn.").replace(
            ".cross_attention.", ".multihead_attn.").replace(".attention.", ".self_attn.")
        mapped[key] = value
    upstream.load_state_dict(mapped, strict=True)
    source = torch.randn(2, 32, 2, 2, requires_grad=True)
    position = torch.randn(1, 32, 2, 2)
    query = torch.randn(4, 32)
    latent, state, language = torch.randn(3, 2, 32, requires_grad=True).unbind(0)
    extra_pos = torch.randn(3, 32)
    actual = local(source, position, query, latent, state, language, extra_pos)
    # Upstream encoder/decoder are unchanged; only add the agreed language token.
    tokens = torch.cat((torch.stack((latent, state, language)), source.flatten(2).permute(2, 0, 1)))
    positions = torch.cat((extra_pos[:, None].expand(-1, 2, -1),
                           position.flatten(2).permute(2, 0, 1).expand(-1, 2, -1)))
    memory = upstream.encoder(tokens, pos=positions)
    queries = query[:, None].expand(-1, 2, -1)
    expected = upstream.decoder(torch.zeros_like(queries), memory, pos=positions,
                                query_pos=queries)[0].transpose(0, 1)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual.square().sum(), (source, language)),
                    torch.autograd.grad(expected.square().sum(), (source, language))):
        torch.testing.assert_close(a, b)


def test_dp_visual_encoder_matches_upstream(monkeypatch):
    checkout = RESEARCH / "diffusion_policy"
    if not checkout.exists():
        pytest.skip("Local upstream checkout unavailable")
    monkeypatch.syspath_prepend(str(checkout))
    from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
    from torchvision.models import resnet18
    from src.policy.dp.model import VisualEncoder
    local = VisualEncoder(2, 32, 28, 512, None).eval()
    upstream = MultiImageObsEncoder(
        {"obs": {key: {"shape": [3, 32, 32], "type": "rgb"} for key in ("a", "b")}},
        resnet18(weights=None), crop_shape=(28, 28), random_crop=True,
        use_group_norm=True, share_rgb_model=False, imagenet_norm=True,
    ).eval()
    # The upstream helper expects a backbone whose classification head is removed.
    for index, key in enumerate(("a", "b")):
        upstream.key_model_map[key].fc = torch.nn.Identity()
        upstream.key_model_map[key].load_state_dict(local.encoders[index].backbone.state_dict())
    images = torch.rand(2, 2, 3, 32, 32)
    torch.testing.assert_close(local(images), upstream({"a": images[:, 0], "b": images[:, 1]}))


def test_dp3_point_encoder_matches_upstream(monkeypatch):
    checkout = RESEARCH / "3D-Diffusion-Policy/3D-Diffusion-Policy"
    if not checkout.exists():
        pytest.skip("Local upstream checkout unavailable")
    monkeypatch.syspath_prepend(str(checkout))
    from diffusion_policy_3d.model.vision.pointnet_extractor import PointNetEncoderXYZ
    from src.policy.dp3.model import DP3Policy
    local = DP3Policy(num_points=8, down_dims=(16, 32))
    upstream = PointNetEncoderXYZ(in_channels=3, out_channels=64,
                                  use_layernorm=True, final_norm="layernorm")
    upstream.mlp.load_state_dict(local.nets["point_mlp"].state_dict())
    upstream.final_projection.load_state_dict(local.nets["point_projection"].state_dict())
    points = torch.randn(2, 8, 3, requires_grad=True)
    actual = local.nets["point_projection"](local.nets["point_mlp"](points).amax(dim=1))
    expected = upstream(points)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), points)[0],
                               torch.autograd.grad(expected.square().sum(), points)[0])
