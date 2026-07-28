# robobrain环境
python -m script.server \
--example libero \
--ckpt-path examples/libero/result/0723/checkpoints/step_30000.pt \
--execute-chunk-len 5 \
--complete-window 2

# libero_10
python -m examples.libero.eval.client \
--task-suite-name libero_10 \
--tasks 6 \
--num-trials-per-task 3 \
--max-steps 950 \
--control-freq 10

# libero_custom
python -m examples.libero.eval.client \
--task-suite-name libero_custom \
--tasks 1 \
--num-trials-per-task 50 \
--max-steps 950 \
--control-freq 10


python -m examples.libero.eval.client_offline \
  --task-suite-name libero_10 \
  --tasks 6 \
  --episodes 0 \
  --sample 65
 

 # libero_custom 任务列表（任务序号从 0 开始）
#
# 0: white mug on → pudding right
#    把白杯放到盘子上，然后把巧克力布丁放到盘子右侧。
#
# 1: pudding right → white mug on
#    把巧克力布丁放到盘子右侧，然后把白杯放到盘子上。
#
# 2: white mug right
#    只把白杯放到盘子右侧。
#
# 3: pudding on
#    只把巧克力布丁放到盘子上。
#
# 4: white mug right → pudding on
#    把白杯放到盘子右侧，然后把巧克力布丁放到盘子上。
#
# 5: white mug left plate → yellow-white mug right plate
#    把白杯放到左侧盘子上，然后把黄白杯放到右侧盘子上。
#
# 6: yellow-white mug right plate → white mug left plate
#    把黄白杯放到右侧盘子上，然后把白杯放到左侧盘子上。
#
# 7: white mug on → pudding right（交换初始位置）
#    任务语义和完成条件与任务 0 相同，但白杯与巧克力布丁的初始区域互换。