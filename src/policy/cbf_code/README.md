# CbF-Code

This baseline adapts the released Compose by Focus implementation to LIBERO. It
keeps its shared DP3-style point encoder, three nodes, fixed directed chain,
two GAT layers, graph mean pooling, frozen CLIP ViT-B/32 text feature plus MLP,
epsilon-prediction DDPM, 16-step horizon, two observations, eight returned
actions. This GraphVLA baseline intentionally trains and evaluates raw weights
without the EMA copy used by the released training script.

The default experiment uses the established `libero_custom_0904_20hz` ID split:
7 training tasks and the first 10 episodes per task (70 episodes total).

The benchmark-specific changes are 32 segmented XYZ points per entity instead
of 100, an 8D LIBERO state, and a 7D LIBERO delta action. Object inputs come
from `node_points_xyz`, not `node_points_xyz_track`. The six analytic gripper
points are repeated to 32 points for the actor node.

The public code passes relationship features to `GATConv` but does not set
`edge_dim`; consequently they do not enter attention. This implementation
intentionally has no semantic edge feature path. This is faithful to the code,
and is not the paper's graph-structure ablation.
