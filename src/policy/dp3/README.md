# Language-conditioned DP3 for LIBERO

Architecture reference: [official DP3](https://github.com/YanjieZe/3D-Diffusion-Policy),
`dp3.yaml` (not Simple DP3). The XYZ PointNet uses 64/128/256 channels with
LayerNorm, max pooling and a 64-dimensional projection. The state MLP is
8/64/64. The shared FiLM U-Net matches the upstream global-condition path,
with 512/1024/2048 channels, x0 prediction, 100 diffusion steps and 10 DDIM
inference steps. The upstream MIT notice is in LICENSE.

Adaptations:

- Use previous and current observations (`obs_steps=2`), with start-of-episode
  history padded by repeating the first frame.
- Frozen BGE-small-en-v1.5 uses L2-normalized CLS pooling. Its 384-dimensional
  embedding is projected to 64 dimensions and concatenated with point/state features.
- Predict 16 delta actions starting at t-1 and return indices 1:11 (t through t+9); no temporal aggregation.
- Mask episode-end padding in the action loss.
- No EMA: inference uses the trained weights directly. The frozen text encoder
  is loaded separately.
- Train with the same 12 tasks, first 10 episodes per task, global batch 64,
  and 30,000 optimizer steps as the current baselines.
- State/actions use the existing suite quantiles, as in the image DP baseline.
  Scene XYZ remains in camera coordinates in meters (no object-point statistics
  or additional point normalization). These data adaptations differ from
  upstream dataset-specific min/max normalization.

Training reads the existing `observation.point_cloud` (512x3). Online inference
back-projects agentview metric depth with camera intrinsics, selects 8192 candidate
pixels via rounded linspace, filters invalid depths and uses deterministic FPS.
This reproduces `src/dataset/lerobot_add_scene_pcd.py` at git revision `a953944`.
There is no world transform, RGB feature, segmentation, or workspace crop.
Use CUDA for the same FPS arithmetic as the generated dataset; CPU tie breaks may differ.

Run inside the configured container and conda environment:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python -m torch.distributed.run \
  --nproc_per_node=1 --master_port=29505 src/training/training.py \
  --example libero --policy dp3
```

Inference (after training; use the actual saved checkpoint):

```bash
CUDA_VISIBLE_DEVICES=0 python -m script.server --example libero \
  --ckpt-path examples/libero/result/dp3_bge_custom0902_obs2/checkpoints/step_30000.pt \
  --execute-chunk-len 10 --port 8008
```

Use `--delta-action --control-freq 20 --port 8008` on the LIBERO client.
Specify the same task IDs, trial count and initial states as the other baselines.
For a new experiment, change `wandb_name`: `resume=True` restores saved configs.
