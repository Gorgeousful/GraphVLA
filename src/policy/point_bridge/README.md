# Point Bridge LIBERO baseline

This is a shared-perception LIBERO adaptation of the official Point Bridge
implementation in `research/pointbridge`, not a reproduction of its sim-to-real
experiments. Both official action representations are available through
`examples/libero/config/point_bridge/model_config.py`:

```python
action_mode: str = "pose"  # default; change to "points" for future robot points
```

The model, data target, output transform, checkpoint name and inference route
follow this setting. Pose and Points checkpoints are incompatible and use
separate default experiment directories.

## Model and language

- Shared PointNet: per-point Linear/LayerNorm/ReLU layers `3 -> 64 -> 128 -> 256`,
  max pooling and a projection to 512 dimensions. Robot points and the flattened
  set of all task object points are encoded separately with shared weights.
- Each observation provides robot and object tokens followed by a learned action
  token. A language token precedes the sequence. The causal GPT has 8 layers,
  4 heads and hidden dimension 256. Default history is one current observation.
- The official deterministic head is a two-layer Transformer decoder with
  learned action queries and causal query attention, not a simple MLP. Its
  unused action-encoder layer is omitted. No diffusion head is included.
- The frozen `all-MiniLM-L6-v2` encoder uses attention-mask mean pooling, L2
  normalization and a 256-token limit, matching the downloaded Sentence
  Transformers configuration. A trainable `384 -> 512 -> 512` ReLU MLP projects
  the sentence embedding. Embeddings are cached and computed in float32; only
  the projector and policy are trained. No BGE substitution or retrieval prefix.
- The code's deterministic loss is fixed-std Gaussian NLL (`stddev=0.1`),
  equivalent to scaled MSE plus a constant. Padded targets are excluded. The
  paper describes this objective as MSE. No progress head or EMA is used.

## Shared data and action contract

- Read existing `node_points_xyz` and `valid_node_mask`, with 32 points per
  object and up to 10 object slots; no perception preprocessing or dataset
  regeneration. Read all task object slots, without role collapse, subtask
  masks, progress labels or point-identity assumptions.
- Use the existing six `gripper_points_xyz` robot points. Invalid object points
  are excluded from PointNet max pooling; an empty object set produces a zero
  feature. Repeated/padded slots cannot dominate max pooling.
- Input and output positions use existing `camera_xyz` q01/q99 statistics,
  mapped to [0,1] without clipping. This replaces the official base-frame
  empirical min/max normalization. No `camera_xyz_track` statistics are used.
- **Pose:** directly predict 10 values per step: camera-frame TCP XYZ, continuous
  rotation 6D (first two rotation-matrix rows), and gripper. Targets come from
  measured `observation.state` at `[t+1,...,t+10]`, transformed with per-task
  `meta/cameras.json`. Rotation components map from [-1,1] to [0,1].
- **Points:** predict the six future robot XYZ points at `[t+1,...,t+10]` and
  gripper repeated over three channels, for 21 values per step. Only robot
  points are supervised; no future object trajectory loss.
- Both heads use gripper commands from `action[t,...,t+9]`, mapped from [-1,1]
  to [0,1]. The dataset's six delta pose channels are not Pose targets. Using
  measured future states was chosen to align Pose with Points; this differs
  from directly supervising controller target commands in the official code.
- Future episode padding is masked; historical episode padding repeats the
  first observation. Pose inference decodes rotation 6D and transforms TCP poses
  to world coordinates. Points inference uses the existing geometry adapter.
  Both execute 7D absolute LIBERO commands with continuous gripper output.

Defaults use the shared 12-task, first-10-demonstrations split (120 episodes),
20 Hz data, batch 64, 30,000 updates, constant AdamW lr 1e-4, weight decay 1e-4,
BF16 when available, and activation checkpointing. Prediction and execution
length default to 10; execution uses a chunk prefix without temporal averaging.
These differ from the official 128 points/object, 40-step chunks at 10 Hz and
temporal averaging. The native GPT, PointNet and deterministic decoder are kept.

## Local language weights

Weights are at `/data0/luokang/dataset/luokang/ckpts/all-MiniLM-L6-v2` (about 88 MiB).
Downloaded from the Hugging Face repository `sentence-transformers/all-MiniLM-L6-v2`
via `hf-mirror.com` because direct Hugging Face DNS failed in the container.
Revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
The directory includes safetensors weights, tokenizer, pooling and sentence
encoder configuration. Loading uses local files only and does not require the
`sentence_transformers` package.

## Training and evaluation

Run inside Docker `lk_cu128`, with conda environment `robobrain` activated,
from the GraphVLA root:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/libero/script/training_point_bridge.sh
```

Default results: `examples/libero/result/point_bridge_pose_custom0902`.
Changing `action_mode` to `"points"` changes this to
`examples/libero/result/point_bridge_points_custom0902`.

```bash
CUDA_VISIBLE_DEVICES=0 python -m script.server \
  --example libero \
  --ckpt-path examples/libero/result/point_bridge_pose_custom0902/checkpoints/step_30000.pt \
  --execute-chunk-len 10 --port 8008 \
  --locator-mode box --keep-locator-loaded

python -m examples.libero.eval.client \
  --task-suite-name libero_custom_0902 \
  --tasks 1 2 3 5 6 8 9 10 12 13 16 17 \
  --num-trials-per-task 50 --max-steps 1000 --control-freq 20 --port 8008
```

For Points, use its checkpoint path; the saved model/data configuration selects
the output adapter automatically. Both modes automatically use the existing
SAM-only mask sampling path; TAPNext is not loaded. The environment controls
success and termination; the server performs no progress-based subtask switch.
Language/reset changes clear per-session perception/history. Do not use
`--delta-action` for either mode.

## Validation

Tests cover both heads, language gradients, permutation invariance, masked and
empty object sets, gradient checkpointing, weight reload, future padding,
pose coordinate/rotation roundtrips, online/offline SAM inputs and server reset.
Both default models have also completed a CPU optimizer step on an actual
LIBERO batch with the local MiniLM encoder and a config/weight snapshot reload.
This does not establish full-run convergence, rollout success or GPU throughput.

Focused regression: 27 passed. Two existing tests were excluded after separately
reproducing their failures without the Point Bridge changes:
`test_environment_controlled_server_and_language_reset` has an outdated mock
missing tracking-response attributes in the current workspace;
`test_delta_release_opens_gripper_then_holds_position` requests delta execution
from the existing absolute-only point adapter. Their code was not changed.

```bash
python -m pytest -q examples/libero/test/test_point_bridge.py \
  examples/libero/test/test_point_policy.py examples/libero/test/test_policy_registry.py \
  examples/libero/test/test_server_action.py \
  -k 'not test_environment_controlled_server_and_language_reset and not test_delta_release_opens_gripper_then_holds_position'
```

Original NVIDIA, nanoGPT and DP3 license notices are included alongside the code.
