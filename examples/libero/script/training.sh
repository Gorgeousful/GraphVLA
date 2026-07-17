CUDA_VISIBLE_DEVICES=0,1 \
WANDB_MODE=offline \
torchrun --nproc_per_node 2 \
src/training/training.py \
--example libero

wandb sync /data0/luokang/research/GraphVLA/examples/libero/result/no_preencoder_point_pos/wandb/latest-run

# 0710 history 16, future 16; abs; weights all 1
# 0714 history 16, future 16; future delta; 0.5 hitory, 0.5 compltete, 0.1 future object
# 0715 history 16, future 8; future delta; 0.5 hitory, 0.5 compltete, 0.0 future object
# 0716 history 16, future 8; future abs; 0.5 hitory, 0.5 compltete, 0.0 future object