# comani-seq
CUDA_VISIBLE_DEVICES=0 \
python -m script.server \
--example libero \
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

#! Seq-OOD
python -m examples.libero.eval.client \
  --task-suite-name libero_custom_0906 \
  --tasks 6 7 8 9 10 11 \
  --num-trials-per-task 10 \
  --max-steps 1000 \
  --control-freq 20 \
  --port 8004 \
  --num-workers 3 \
  --switch-mode oracle
  
#! Seq-ID
python -m examples.libero.eval.client \
  --task-suite-name libero_custom_0906 \
  --tasks 0 1 2 3 4 5 \
  --num-trials-per-task 10 \
  --max-steps 1000 \
  --control-freq 20 \
  --port 8004 \
  --num-workers 3 \
  --switch-mode predicted