# Official image U-Net Diffusion Policy with BGE

Reference: local `research/diffusion_policy`, `DiffusionUnetImagePolicy`,
`MultiImageObsEncoder`, and `train_diffusion_unet_image_workspace.yaml`.
Each camera has a separate ResNet18, randomly initialized, with GroupNorm
and global average pooling (512D). Images are resized to 224x224; crop size is also 224x224 (no spatial cropping), then ImageNet-normalized.
No DROID spatial softmax, color jitter, ResNet50 or repeated noise sampling remains.

Two visual/state observations are flattened. One frozen BGE-small-en-v1.5
full-instruction embedding (L2-normalized CLS, 384D, no retrieval prefix) is
concatenated to that observation vector as U-Net global conditioning.
The FiLM U-Net has channels 512/1024/2048, timestep embedding 128, kernel 5,
and 8 groups. DDPM uses 100 training and 100 inference steps with cosine betas,
fixed-small variance, clipping and epsilon prediction. One noise/timestep is
sampled per training example. Horizon=16; return 10 actions starting at t.

LIBERO adaptations: state=8, delta action=7, existing suite quantile normalization
and padding-masked MSE. Keep the common training framework, configured optimizer
schedule without gradient clipping, AMP and gradient checkpointing; no EMA (an explicit experiment choice).
The shared U-Net implements the upstream global-condition path, without unused
local conditioning/inpainting branches. This is not the hybrid/Transformer variant.

New output directory: `examples/libero/result/dp_official_bge_custom0902`.
Old DROID/DistilBERT checkpoints are incompatible; start a new run.
