python -m script.server \
--example libero \
--ckpt-path examples/libero/result/0811-contactprofile-basetcpfinger-cls4-rolechain-current-progress-sam-7/checkpoints/step_30000.pt \
--execute-chunk-len 5 \
--progress-window 2 \
--progress-threshold 0.85 \
--locator-scale 2.0 \
--port 8001 \
--devices '{"inference":"cuda:0","node_segmenter":"cuda:0","point_tracker":"cuda:0","node_locator":"cuda:1"}' \
--sam-only \
--locator-mode box


# --sam-only
# --locator-mode box 
# --trials-init-state 0 1 2 3 4 5 6 8 10 11


# libero_10 # 1 6 4
python -m examples.libero.eval.client \
--task-suite-name libero_10 \
--tasks 6 \
--num-trials-per-task 10 \
--max-steps 1000 \
--control-freq 10 \
--port 8001

# libero_swap_test
python -m examples.libero.eval.client \
--task-suite-name libero_swap_test \
--tasks 3 \
--num-trials-per-task 10 \
--trials-init-state 10 11 \
--max-steps 1000 \
--control-freq 10 \
--port 8001

# libero_custom
python -m examples.libero.eval.client \
--task-suite-name libero_custom \
--tasks 1 \
--num-trials-per-task 3 \ 
--max-steps 750 \
--control-freq 10
 

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

# libero_swap_test（顺序与 libero_with_depth_0_5_6_7_8 的 task_index 0-4 对齐）
# 0: moka pot on stove -> turn stove on
# 1: tomato sauce -> alphabet soup
# 2: butter -> cream cheese box
# 3: chocolate pudding right -> white mug on plate
# 4: yellow-white mug right plate -> white mug left plate
# python -m examples.libero.eval.client \
# --task-suite-name libero_swap_test \
# --tasks 0,1,2,3,4 \
# --num-trials-per-task 3 \
# --max-steps 750 \
# --control-freq 10
