# expr0: online input vs dataset GT and point-order sensitivity

Sample: episode 0, local sample 28 (global index 28)
Checkpoint: step_15000.pt

## Files

- server_model_input.npz/json: exact tensors captured at PointQueryModel.infer
- dataset_gt_model_input.npz: corresponding transformed dataset tensors
- input_comparison.json/png: field-wise online-vs-GT comparison
- point_order_experiment.json: control and patient-point-permutation metrics
- point_order_predictions.npz: raw and display-space predictions
- future_action_comparison.png: GT, control, and permuted future gripper points
- server_offline/: legacy frame13-reset full-pipeline artifacts
- warmup/: fixed frame0-to-frame28 warm-up capture, comparison, and visualizations

## Reproduce the online capture

Start the server:

    python -m script.server \
      --example libero \
      --ckpt-path examples/libero/result/0716/checkpoints/step_15000.pt \
      --complete-window 100 \
      --port 8001 \
      --devices '{"inference":"cuda:0","node_locator":"cuda:0","sam3":"cuda:1","point_tracker":"cuda:1","depth_predictor":"cuda:1"}'

Capture sample 28 with the legacy frame13 reset:

    python -m examples.libero.eval.client_offline \
      --host 127.0.0.1 --port 8001 \
      --episode-index 0 --sample-index 28 --num-samples 1 \
      --no-warmup-from-episode-start \
      --future-object false \
      --output-dir examples/libero/test/expr0/server_offline \
      --model-input-output examples/libero/test/expr0/server_model_input.npz

The fixed behavior is now the default:

    python -m examples.libero.eval.client_offline \
      --host 127.0.0.1 --port 8001 \
      --episode-index 0 --sample-index 28 --num-samples 1 \
      --future-object false \
      --output-dir examples/libero/test/expr0/warmup/server_offline \
      --model-input-output examples/libero/test/expr0/warmup/server_model_input.npz

This sends episode frames 0 through 28 to warm up STream3R, SAM table-mask
state, and the raw-depth calibrator. The server still keeps only frames 13
through 28 in the 16-frame model feature window.

## Reproduce the GT and permutation experiment

    python examples/libero/test/expr0/run_expr0.py \
      --episode-index 0 --sample-index 28 \
      --server-input examples/libero/test/expr0/server_model_input.npz \
      --ckpt-path examples/libero/result/0716/checkpoints/step_15000.pt \
      --output-dir examples/libero/test/expr0 \
      --device cuda:0 --seed 0

The experimental input applies one seeded permutation to the 32 patient/mug
point indices and reuses that permutation for all 16 history frames. Actor
features, target/plate points, condition embeddings, query IDs, and targets are
unchanged.

## Key results

- Control future actor UV error: 4.9176 px mean.
- Patient-permuted future actor UV error: 9.5965 px mean.
- Permuted-vs-control prediction shift: 7.1695 px mean, 17.0053 px max.
- Online-vs-GT actor UV and gripper metric are effectively identical.
- Online-vs-GT actor depth norm MAE is 0.2192 (max 0.7191).
- Online-vs-GT unordered mug UV Chamfer is 3.5210 px; aligned point-index UV
  errors are much larger, showing a point-index/order distribution mismatch.

## Episode warm-up comparison

Compared with the legacy frame13 reset:

| Metric | Legacy | Frame0 warm-up |
| --- | ---: | ---: |
| Actor depth norm MAE | 0.219176 | 0.000003 |
| Mug depth norm MAE | 0.172390 | 0.069885 |
| Plate depth norm MAE | 0.070334 | 0.039701 |
| Future gripper absolute error | 0.031234 | 0.036451 |

The actor/gripper input depth now matches dataset GT to numerical precision.
The remaining future-gripper error is model-output error rather than an input
anchor mismatch. Mug and plate retain smaller residual discrepancies from the
online segmentation/tracking path.
