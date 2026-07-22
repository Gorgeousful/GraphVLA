# robobrain环境
python -m script.server \
--example libero \
--ckpt-path examples/libero/result/0722_fix/checkpoints/step_10000.pt \
--complete-window 3 \
--devices '{"inference":"cuda:0","node_locator":"cuda:0","sam3":"cuda:1","point_tracker":"cuda:1","depth_predictor":"cuda:1"}'

# libero环境
python -m examples.libero.eval.client \
--task-suite-name libero_10 \
--tasks 6 \
--num-trials-per-task 1 \
--max-steps 150 \
--execute-chunk-len 1

python -m examples.libero.eval.client \
--task-suite-name libero_10_swap \
--tasks 1 \
--num-trials-per-task 1 \
--max-steps 150 \
--execute-chunk-len 5


python -m examples.libero.eval.client_offline \
  --task-suite-name libero_10 \
  --tasks 6 \
  --episodes 0 \
  --sample 65 \
  --execute-chunk-len 5