CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=offline \
torchrun --nproc_per_node 1 \
--master_port 29501 \
src/training/training.py \
--example libero \
--policy act

CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=offline \
torchrun --nproc_per_node 1 \
--master_port 29504 \
src/training/training.py \
--example libero \
--policy dp

CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=offline \
torchrun --nproc_per_node 1 \
--master_port 29505 \
src/training/training.py \
--example libero \
--policy dp3

CUDA_VISIBLE_DEVICES=1 \
WANDB_MODE=offline \
torchrun --nproc_per_node 1 \
--master_port 29506 \
src/training/training.py \
--example libero \
--policy point_policy

CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=offline \
torchrun --nproc_per_node 1 \
--master_port 29506 \
src/training/training.py \
--example libero \
--policy point_bridge

CUDA_VISIBLE_DEVICES=1 \
WANDB_MODE=offline \
torchrun --nproc_per_node 1 \
--master_port 29502 \
src/training/training.py \
--example libero \
--policy graphpoint

CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=offline \
torchrun --nproc_per_node 1 \
--master_port 29503 \
src/training/training.py \
--example libero \
--policy graphpoint_gc
