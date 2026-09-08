# 正式版本
#! OOD
CUDA_VISIBLE_DEVICES=1 \
python -m script.server \
--example libero \
--ckpt-path examples/libero/result/0906-pointdropknn-basetcpfinger-cls4-rolechain-current-progress-sam-custom0902-ep10-rel/checkpoints/step_30000.pt \
--execute-chunk-len 10 \
--progress-window 2 \
--progress-threshold 0.9 \
--locator-scale 2.0 \
--port 8003 \
--devices '{"inference":"cuda:0","node_segmenter":"cuda:0","point_tracker":"cuda:0","node_locator":"cuda:0"}' \
--sam-only \
--locator-mode box \
--keep-locator-loaded

python -m examples.libero.eval.client \
--task-suite-name libero_custom_0902 \
--tasks 0 4 7 11 14 15 \
--num-trials-per-task 50 \
--max-steps 1000 \
--control-freq 20 \
--port 8003 \
--num-workers 3

#! ID
CUDA_VISIBLE_DEVICES=0 \
python -m script.server \
--example libero \
--ckpt-path examples/libero/result/0906-pointdropknn-basetcpfinger-cls4-rolechain-current-progress-sam-custom0902-ep10-rel/checkpoints/step_30000.pt \
--execute-chunk-len 10 \
--progress-window 2 \
--progress-threshold 0.9 \
--locator-scale 2.0 \
--port 8002 \
--devices '{"inference":"cuda:0","node_segmenter":"cuda:0","point_tracker":"cuda:0","node_locator":"cuda:0"}' \
--sam-only \
--locator-mode box \
--keep-locator-loaded

python -m examples.libero.eval.client \
--task-suite-name libero_custom_0902 \
--tasks 1 2 3 5 6 8 9 10 12 13 16 17  \
--num-trials-per-task 50 \
--max-steps 1000 \
--control-freq 20 \
--port 8002 \
--num-workers 3 \
--resume-dir examples/libero/eval/output/0906-pointdropknn-basetcpfinger-cls4-rolechain-current-progress-sam-custom0902-ep10-rel-step_30000/libero_custom_0902-0907-1508