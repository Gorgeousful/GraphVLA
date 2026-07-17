python examples/libero/test/client_offline.py \
--port 10092


# robobrain环境
python -m script.server \
--example libero \
--ckpt-path examples/libero/result/0716/checkpoints/step_15000.pt \
--complete-window 100 \
--devices '{"inference":"cuda:0","node_locator":"cuda:0","sam3":"cuda:1","point_tracker":"cuda:1","depth_predictor":"cuda:1"}'

# libero环境
python -m examples.libero.eval.client \
--task-suite-name libero_10 \
--tasks 6 \
--num-trials-per-task 1 \
--max-steps 300 \
--execute-chunk-len 8 \
--future-object false

# offline测试
python -m examples.libero.eval.client_offline \
--episode-index 0 \
--sample-index 28 \
--num-samples 1