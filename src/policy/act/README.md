# Official ACT with BGE language conditioning

Reference: local `research/act`, `detr/models/{backbone,detr_vae,transformer}.py`.
ResNet18 (ImageNet-1K weights, FrozenBatchNorm) is shared across cameras.
The CVAE posterior consumes CLS, state and action tokens; inference uses z=0.
BGE-small-en-v1.5 is frozen. Full instructions have no retrieval prefix;
L2-normalized CLS (384D) is projected to a separate Transformer encoder token.
There is no EfficientNet or language FiLM in the image backbone.

LIBERO adaptations: state=8, delta action=7, dual RGB resized to 224x224,
chunk=10, suite mean/std normalization, padding-masked L1 plus 10*KL.
Like upstream DETRVAE, predictions use decoder output index 0 (the first
layer), even when seven layers are configured. Later decoder layers therefore
have no action-loss contribution. The unused padding prediction head is omitted.
The existing training schedule, AMP and gradient checkpointing are retained.
No EMA is used. This is an official-architecture adaptation, not an unchanged
reproduction of the original ALOHA training protocol.

New output directory: `examples/libero/result/act_official_bge_custom0902`.
Old EfficientNet/DistilBERT weights are incompatible; start a new run.
