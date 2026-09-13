# Point-Policy LIBERO baseline

This is a language-conditioned LIBERO adaptation of the deterministic actor in
`research/Point-Policy/point_policy/agent/point_policy.py`, not an exact reproduction
of its Franka experiments. The original MIT notices are included in this folder.

## Model and data contract

- Each point's XYZ history is flattened and projected to 512 dimensions. A
  non-causal GPT with hidden dimension 256, four layers and two heads processes
  the point tokens. A shared two-hidden-layer ReLU head predicts future XYZ.
- Frozen BGE-small-en-v1.5 CLS features are L2-normalized (384 dimensions), then
  projected to 256 dimensions as one additional language token. Full instructions
  are encoded without a retrieval prefix, in float32 in both training and inference.
  This is a multi-task adaptation; the original Point-Policy actor has no text input.
- Inputs use `node_points_xyz_track`, generated with TAPNext, rather than the
  independently sampled SAM points in `node_points_xyz`. The six robot points
  come from `gripper_points_xyz`. Object slots keep their dataset ordering.
- Ten observations at offsets `[-9,...,0]` are used, matching the current
  GraphPoint window. All valid task object slots are supplied, including objects
  belonging to later subtasks. No `subtask_id`, `subtask_node_mask` or progress
  labels are needed. There is no role-chain attention or point dropout.
- The positional embedding capacity is enlarged from the original 20-token
  limit to 328 (6 robot + 10*32 object + gripper + language). Invalid object
  slots are masked in attention; histories are zeroed only where invalid.
- Camera-frame XYZ uses the existing `camera_xyz_track` q01/q99 statistics,
  mapped to [0,1] without clipping. These stats include tracked object and robot
  points. The original repository uses empirical min/max instead; we use the
  dataset's available tracked-point statistics consistently at train and eval.
- Targets are robot points at `[t+1,...,t+10]`. Gripper commands at `[t,...,t+9]`
  are mapped from [-1,1] to [0,1] and repeated over three output channels.
  Observation gripper closedness comes from measured finger-point separation.
  Episode-edge observations repeat the first frame; padded future labels are
  excluded from the loss. Object trajectories are conditioning only.
- The loss is fixed-std Gaussian NLL (`stddev=0.1`), as in the original
  deterministic head. It can be negative. No progress head or EMA is used.
- Predictions are restored to camera coordinates and converted to absolute
  LIBERO poses through the shared geometry adapter. The gripper's three outputs
  are averaged and mapped back to [-1,1].

Defaults are batch 64, 30,000 updates, constant AdamW lr 1e-4, weight decay 1e-4,
betas (0.9,0.999), no gradient clipping, BF16 when available, and activation
checkpointing enabled. The task list matches the other baseline configs: 12
selected tasks, the first ten demonstrations each (120 episodes), not all 18
tasks in the source dataset. No existing dataset files are modified.

## Training

Inside `lk_cu128`, activate `robobrain` and run from the GraphVLA repository root:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/libero/script/training_point_policy.sh
```

The public policy name is `point_policy`. Configs live in
`examples/libero/config/point_policy/`; results default to
`examples/libero/result/point_policy_custom0902`.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m script.server \
  --example libero \
  --ckpt-path examples/libero/result/point_policy_custom0902/checkpoints/step_30000.pt \
  --execute-chunk-len 10 --port 8007 \
  --locator-mode box --keep-locator-loaded
```

Do not pass `--sam-only`: initial SAM points must be tracked with TAPNext across
frames. The online projection preserves point identity and matches the stored
tracked-point projection: all in-bounds, positive-finite-depth points are used,
including low-visibility predictions, because per-point visibility was not saved.

In a separate terminal, use the existing LIBERO client in absolute-action mode
(its default; do not pass `--delta-action`):

```bash
python -m examples.libero.eval.client \
  --task-suite-name libero_custom_0902 \
  --tasks 1 2 3 5 6 8 9 10 12 13 16 17 \
  --num-trials-per-task 50 --max-steps 1000 --control-freq 20 --port 8007
```

The environment determines success and termination. The server does not perform
progress-based subtask switching. Language/reset changes clear tracking history.
Unlike the original default evaluation, this adaptation executes a configured
prefix of each predicted chunk and does not apply temporal aggregation, matching
the other chunk-execution baselines here. Prediction length and execution length
are separate settings.

Validation covers tracked data, language gradients, mask/padding semantics,
activation checkpointing, snapshot/spawn loading, weight roundtrip, and
train/inference projection and normalization. It does not establish rollout
success or throughput with the full online perception stack.
