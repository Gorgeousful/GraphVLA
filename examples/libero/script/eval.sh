python examples/libero/test/client_offline.py \
--port 10092


# robobrain环境
python -m script.server \
--example libero \
--ckpt-path examples/libero/result/checkpoints/step_10000.pt \
--complete-window 100 \
--devices '{"inference":"cuda:0","node_locator":"cuda:0","sam3":"cuda:1","point_tracker":"cuda:1","depth_predictor":"cuda:1"}'

# libero环境
python -m examples.libero.eval.client \
--task-suite-name libero_10 \
--tasks 6 \
--num-trials-per-task 1 \
--max-steps 250 \
--execute-chunk-len 4 