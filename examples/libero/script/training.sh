CUDA_VISIBLE_DEVICES=1,2,5,6 \
WANDB_MODE=offline \
torchrun --nproc_per_node 4 \
src/training/training.py \
--example libero

CUDA_VISIBLE_DEVICES=0,1 \
WANDB_MODE=offline \
torchrun --nproc_per_node 2 \
--master_port 29502 \
src/training/training.py \
--example libero