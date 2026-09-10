# 对比实验
#? act
CUDA_VISIBLE_DEVICES=0 \
python -m script.server \
  --example libero \
  --ckpt-path examples/libero/result/act_official_bge_custom0902/checkpoints/step_30000.pt \
  --execute-chunk-len 10 \
  --device cuda:0 \
  --port 8002

#! OOD
python -m examples.libero.eval.client \
  --task-suite-name libero_custom_0902 \
  --tasks 0 4 7 11 14 15 \
  --num-trials-per-task 50 \
  --max-steps 1000 \
  --control-freq 20 \
  --delta-action \
  --port 8002 \
  --num-workers 3

#! ID
python -m examples.libero.eval.client \
  --task-suite-name libero_custom_0902 \
  --tasks 1 2 3 5 6 8 9 10 12 13 16 17  \
  --num-trials-per-task 50 \
  --max-steps 1000 \
  --control-freq 20 \
  --delta-action \
  --port 8002 \
  --num-workers 3

#? dp
CUDA_VISIBLE_DEVICES=0 \
python -m script.server \
  --example libero \
  --ckpt-path examples/libero/result/dp_official_bge_custom0902/checkpoints/step_30000.pt \
  --execute-chunk-len 10 \
  --device cuda:0 \
  --port 8002

#! OOD
python -m examples.libero.eval.client \
  --task-suite-name libero_custom_0902 \
  --tasks 0 4 7 11 14 15 \
  --num-trials-per-task 50 \
  --max-steps 1000 \
  --control-freq 20 \
  --delta-action \
  --port 8002 \
  --num-workers 3

#! ID
python -m examples.libero.eval.client \
  --task-suite-name libero_custom_0902 \
  --tasks 1 2 3 5 6 8 9 10 12 13 16 17  \
  --num-trials-per-task 50 \
  --max-steps 1000 \
  --control-freq 20 \
  --delta-action \
  --port 8002 \
  --num-workers 3

#? dp3
CUDA_VISIBLE_DEVICES=0 \
python -m script.server \
  --example libero \
  --ckpt-path examples/libero/result/dp3_bge_custom0902_obs2/checkpoints/step_30000.pt \
  --execute-chunk-len 10 \
  --device cuda:0 \
  --port 8002

#! OOD
python -m examples.libero.eval.client \
  --task-suite-name libero_custom_0902 \
  --tasks 0 4 7 11 14 15 \
  --num-trials-per-task 50 \
  --max-steps 1000 \
  --control-freq 20 \
  --delta-action \
  --port 8002 \
  --num-workers 3

#! ID
python -m examples.libero.eval.client \
  --task-suite-name libero_custom_0902 \
  --tasks 1 2 3 5 6 8 9 10 12 13 16 17 \
  --num-trials-per-task 50 \
  --max-steps 1000 \
  --control-freq 20 \
  --delta-action \
  --port 8002 \
  --num-workers 3

#? point_policy
CUDA_VISIBLE_DEVICES=0 \
python -m script.server \
  --example libero \
  --ckpt-path examples/libero/result/point_policy_custom0902/checkpoints/step_30000.pt \
  --execute-chunk-len 10 \
  --device cuda:0 \
  --locator-scale 2.0 \
  --locator-mode box \
  --keep-locator-loaded \
  --port 8002

#! OOD
python -m examples.libero.eval.client \
  --task-suite-name libero_custom_0902 \
  --tasks 0 4 7 11 14 15 \
  --num-trials-per-task 50 \
  --max-steps 1000 \
  --control-freq 20 \
  --absolute-action \
  --port 8002 \
  --num-workers 3

#! ID
python -m examples.libero.eval.client \
  --task-suite-name libero_custom_0902 \
  --tasks 1 2 3 5 6 8 9 10 12 13 16 17 \
  --num-trials-per-task 50 \
  --max-steps 1000 \
  --control-freq 20 \
  --absolute-action \
  --port 8002 \
  --num-workers 3

