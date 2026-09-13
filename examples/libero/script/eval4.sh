#? embodiment and points vs delta-action
# Run server/client blocks in separate terminals; UR5e and Sawyer reuse port 8004.
#? points - ur5e =====================================================================================================
CUDA_VISIBLE_DEVICES=0 \
python -m script.server \
--example libero \
--embodiment ur5e \
--ckpt-path examples/libero/result/0906-pointdropknn005-basetcpfinger-cls4-rolechain-current-progress-sam-custom0902-ep10-rel/checkpoints/step_30000.pt \
--execute-chunk-len 10 \
--progress-window 2 \
--progress-threshold 0.9 \
--locator-scale 2.0 \
--port 8004 \
--devices '{"inference":"cuda:0","node_segmenter":"cuda:0","point_tracker":"cuda:0","node_locator":"cuda:0"}' \
--sam-only \
--locator-mode box \
--keep-locator-loaded

#! OOD
python -m examples.libero.eval.client \
  --embodiment ur5e \
  --absolute-action \
  --task-suite-name libero_custom_0902 \
  --tasks 0 4 7 11 14 15 \
  --num-trials-per-task 30 \
  --max-steps 1000 \
  --control-freq 20 \
  --port 8004 \
  --num-workers 3
  
#! ID
python -m examples.libero.eval.client \
  --embodiment ur5e \
  --absolute-action \
  --task-suite-name libero_custom_0902 \
  --tasks 1 2 3 5 6 8 9 10 12 13 16 17 \
  --num-trials-per-task 30 \
  --max-steps 1000 \
  --control-freq 20 \
  --port 8004 \
  --num-workers 3

#? points - sawyer =====================================================================================================
CUDA_VISIBLE_DEVICES=0 \
python -m script.server \
--example libero \
--embodiment sawyer \
--ckpt-path examples/libero/result/0906-pointdropknn005-basetcpfinger-cls4-rolechain-current-progress-sam-custom0902-ep10-rel/checkpoints/step_30000.pt \
--execute-chunk-len 10 \
--progress-window 2 \
--progress-threshold 0.9 \
--locator-scale 2.0 \
--port 8004 \
--devices '{"inference":"cuda:0","node_segmenter":"cuda:0","point_tracker":"cuda:0","node_locator":"cuda:0"}' \
--sam-only \
--locator-mode box \
--keep-locator-loaded

#! OOD
python -m examples.libero.eval.client \
  --embodiment sawyer \
  --absolute-action \
  --task-suite-name libero_custom_0902 \
  --tasks 0 4 7 11 14 15 \
  --num-trials-per-task 30 \
  --max-steps 1000 \
  --control-freq 20 \
  --port 8004 \
  --num-workers 3

#! ID
python -m examples.libero.eval.client \
  --embodiment sawyer \
  --absolute-action \
  --task-suite-name libero_custom_0902 \
  --tasks 1 2 3 5 6 8 9 10 12 13 16 17 \
  --num-trials-per-task 30 \
  --max-steps 1000 \
  --control-freq 20 \
  --port 8004 \
  --num-workers 3

# Delta-action checkpoint is still training; run after step_30000.pt is available.

#? delta-action - franka_panda =====================================================================================================
CUDA_VISIBLE_DEVICES=0 \
python -m script.server \
--example libero \
--embodiment franka_panda \
--ckpt-path /data0/luokang/research/GraphVLA/examples/libero/result/0906-pointdropknn005-basetcpfinger-cls4-rolechain-current-progress-sam-custom0902-ep10-relaction/checkpoints/step_30000.pt \
--execute-chunk-len 10 \
--progress-window 2 \
--progress-threshold 0.9 \
--locator-scale 2.0 \
--port 8004 \
--devices '{"inference":"cuda:0","node_segmenter":"cuda:0","point_tracker":"cuda:0","node_locator":"cuda:0"}' \
--sam-only \
--locator-mode box \
--keep-locator-loaded

#! OOD
python -m examples.libero.eval.client \
  --embodiment franka_panda \
  --delta-action \
  --task-suite-name libero_custom_0902 \
  --tasks 0 4 7 11 14 15 \
  --num-trials-per-task 30 \
  --max-steps 1000 \
  --control-freq 20 \
  --port 8004 \
  --num-workers 3

#! ID
python -m examples.libero.eval.client \
  --embodiment franka_panda \
  --delta-action \
  --task-suite-name libero_custom_0902 \
  --tasks 1 2 3 5 6 8 9 10 12 13 16 17 \
  --num-trials-per-task 30 \
  --max-steps 1000 \
  --control-freq 20 \
  --port 8004 \
  --num-workers 3

#? delta-action - ur5e =====================================================================================================
CUDA_VISIBLE_DEVICES=0 \
python -m script.server \
--example libero \
--embodiment ur5e \
--ckpt-path /data0/luokang/research/GraphVLA/examples/libero/result/0906-pointdropknn005-basetcpfinger-cls4-rolechain-current-progress-sam-custom0902-ep10-relaction/checkpoints/step_30000.pt \
--execute-chunk-len 10 \
--progress-window 2 \
--progress-threshold 0.9 \
--locator-scale 2.0 \
--port 8004 \
--devices '{"inference":"cuda:0","node_segmenter":"cuda:0","point_tracker":"cuda:0","node_locator":"cuda:0"}' \
--sam-only \
--locator-mode box \
--keep-locator-loaded

#! OOD
python -m examples.libero.eval.client \
  --embodiment ur5e \
  --delta-action \
  --task-suite-name libero_custom_0902 \
  --tasks 0 4 7 11 14 15 \
  --num-trials-per-task 30 \
  --max-steps 1000 \
  --control-freq 20 \
  --port 8004 \
  --num-workers 3

#! ID
python -m examples.libero.eval.client \
  --embodiment ur5e \
  --delta-action \
  --task-suite-name libero_custom_0902 \
  --tasks 1 2 3 5 6 8 9 10 12 13 16 17 \
  --num-trials-per-task 30 \
  --max-steps 1000 \
  --control-freq 20 \
  --port 8004 \
  --num-workers 3

#? delta-action - sawyer =====================================================================================================
CUDA_VISIBLE_DEVICES=0 \
python -m script.server \
--example libero \
--embodiment sawyer \
--ckpt-path /data0/luokang/research/GraphVLA/examples/libero/result/0906-pointdropknn005-basetcpfinger-cls4-rolechain-current-progress-sam-custom0902-ep10-relaction/checkpoints/step_30000.pt \
--execute-chunk-len 10 \
--progress-window 2 \
--progress-threshold 0.9 \
--locator-scale 2.0 \
--port 8004 \
--devices '{"inference":"cuda:0","node_segmenter":"cuda:0","point_tracker":"cuda:0","node_locator":"cuda:0"}' \
--sam-only \
--locator-mode box \
--keep-locator-loaded

#! OOD
python -m examples.libero.eval.client \
  --embodiment sawyer \
  --delta-action \
  --task-suite-name libero_custom_0902 \
  --tasks 0 4 7 11 14 15 \
  --num-trials-per-task 30 \
  --max-steps 1000 \
  --control-freq 20 \
  --port 8004 \
  --num-workers 3

#! ID
python -m examples.libero.eval.client \
  --embodiment sawyer \
  --delta-action \
  --task-suite-name libero_custom_0902 \
  --tasks 1 2 3 5 6 8 9 10 12 13 16 17 \
  --num-trials-per-task 30 \
  --max-steps 1000 \
  --control-freq 20 \
  --port 8004 \
  --num-workers 3
