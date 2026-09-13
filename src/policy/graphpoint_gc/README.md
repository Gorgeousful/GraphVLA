# GraphPoint-GC

Copied from the working GraphPoint implementation for the single-node ablation.
Actor, patient, and target points are merged before the first attention layer.
There are no role embeddings, role-specific local groups, or chain masks.
Actor keypoint embeddings and the flow decoder's actor history remain intact.

`cls_token_num` is the **total** shared summary count: the default is 12,
matching GraphPoint's three groups of four. Local attention processes all points
within each frame; global attention retains the original temporal positions and
register/dense layer options. Missing roles mask points, never shared summaries.

Progress reads all current shared summaries. For tasks without a target, a
second, progress-only encoder pass adds the subtask-initial patient points to
each frame's shared set using the otherwise vacant target input slot. These
reference points do not enter the flow memory or actor history. Progress loss
and flow objectives are unchanged. Role-selective `progresshead_input` and
`node_attention_mode` are removed from this baseline's configuration.

At the copied default settings, GraphPoint has 150,088,724 parameters and GC has
151,140,372 (+0.70%), mainly because progress reads 12 rather than 8 summaries.
Merged local attention also increases compute; equal token counts do not imply
equal FLOPs. This ablates role-separated encoding, not all robot identity or
task-aware point selection in the input pipeline.

Select `--policy graphpoint_gc` with the existing training entry point.
Independent configs live in `examples/libero/config/graphpoint_gc/`, with a
separate experiment name and `resume=False`. The server resolves the policy
from its saved config. Train fresh; GraphPoint checkpoints are not compatible.
