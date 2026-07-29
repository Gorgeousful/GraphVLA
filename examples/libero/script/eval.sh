# robobrain环境 0723
python -m script.server \
--example libero \
--ckpt-path examples/libero/result/0728/checkpoints/step_150000.pt \
--execute-chunk-len 5 \
--complete-window 1

# libero_10
python -m examples.libero.eval.client \
--task-suite-name libero_10 \
--tasks 6 \
--num-trials-per-task 3 \
--max-steps 750 \
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

# libero_10
# 0	Put both the alphabet soup and the tomato sauce in the basket
# 1	Put both the cream cheese box and the butter in the basket
# 2	Turn on the stove and put the moka pot on it
# 3	Put the black bowl in the bottom drawer of the cabinet and close it
# 4	Put the white mug on the left plate and put the yellow and white mug on the right plate
# 5	Pick up the book and place it in the back compartment of the caddy
# 6	Put the white mug on the plate and put the chocolate pudding to the right of the plate
# 7	Put both the alphabet soup and the cream cheese box in the basket
# 8	Put both moka pots on the stove
# 9	Put the yellow and white mug in the microwave and close it