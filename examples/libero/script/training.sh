CUDA_VISIBLE_DEVICES=0,1 \
WANDB_MODE=offline \
torchrun --nproc_per_node 2 \
src/training/training.py \
--example libero

wandb sync /data0/luokang/research/GraphVLA/examples/libero/result/0728/wandb/latest-run
