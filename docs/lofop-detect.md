# LOFOP-Detect: Design Document

LOFOP-Detect is the flagship detector of the LOFOP framework. This document records the concept
study behind it, every architectural decision with its trade-offs, and how it differs -- on
purpose -- from its research baseline.

## 1. Concept study: what RT-DETR teaches (and what we take)

RT-DETR (Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection") is the research baseline
for LOFOP-Detect. We study its *ideas*; we do not copy its code or recreate its architecture.
Four ideas from the paper matter to us:

1. **Attention only where it is cheap.** RT-DETR's hybrid encoder applies self-attention only to
   the lowest-resolution feature map (stride 32) and uses convolutions for cross-scale fusion,
   because attention cost is quadratic in token count. *Adopted as a principle* in our neck:
   global attention on the stride-32 map only, convolutional fusion everywhere else.
2. **Quality-aware scoring.** RT-DETR selects decoder queries by IoU-aware scores so that
   classification confidence agrees with localization quality. *Adopted as a principle*: our head
   has an explicit quality branch trained to predict IoU, and inference scores are the geometric
   mean of classification and quality.
3. **NMS-free decoding via one-to-one matching.** RT-DETR's Hungarian matching removes NMS.
   *Deliberately NOT adopted* -- see the head decision below for why.
4. **Training strategy.** Strong augmentation schedules, EMA weights, and cosine learning-rate
   decay carry much of the result. *Adopted for Phase 4* (training engine), not baked into the
   model.

## 2. Architecture

```
image (B,3,H,W)
  -> RidgeNet backbone      C3 (H/8)   C4 (H/16)   C5 (H/32)
  -> DeltaFusion neck       P3 (H/8)   P4 (H/16)   P5 (H/32)   [attention on P5 only]
  -> ApexHead               per-level: class logits, ltrb distances, quality logit
  -> decode + C++ batched NMS (inference)  /  DynamicTopKAssigner + losses (training)
```

Every block is a registry entry (`backbone/RidgeNet`, `neck/DeltaFusion`, `head/ApexHead`,
`model/LofopDetect`), so each is swappable from YAML without touching the others.

### 2.1 Backbone: RidgeNet

**What:** a stem that downsamples 4x, then four stages of `RidgeBlock`s at strides 4/8/16/32.
A `RidgeBlock` is a residual inverted bottleneck: 7x7 depthwise convolution (spatial mixing) ->
pointwise expansion with SiLU (channel mixing) -> pointwise projection.

**Why it exists:** large depthwise kernels give small-object-friendly receptive fields at a
fraction of the FLOPs of dense 3x3 stacks, and the inverted-bottleneck shape is the best
FLOPs/accuracy trade currently known for CNNs.
**Advantages:** cheap large receptive field; exports cleanly (plain convs); width/depth scaling
gives a model family (`n`/`s`/`m`) from one implementation.
**Disadvantages:** depthwise convs have lower arithmetic intensity than dense convs, so GPU
utilization is worse than a ResNet at equal FLOPs; no pretrained weights yet (must train from
scratch until we publish checkpoints).
**Performance notes:** channels-last memory format and fused SiLU help on GPU; the 7x7 depthwise
is the layer to watch in TensorRT profiles.

### 2.2 Neck: DeltaFusion

**What:** 1x1 lateral projections bring C3/C4/C5 to one width. The stride-32 map passes through
`PulseAttention` -- single-head global self-attention over its (H/32 x W/32) tokens. Then a
top-down pass (nearest upsample + learnable softmax fusion gates) and a bottom-up pass (stride-2
conv + gates) produce P3/P4/P5.

**Why it exists:** small objects need fine resolution (P3) with semantic context from deeper
levels; large objects need P5 with global context. The two passes move information both ways; the
attention block injects image-level context exactly where tokens are fewest (RT-DETR's lesson).
**Advantages:** O(1) attention cost relative to image area growth at P3 (attention never touches
the big maps); learnable gates let training decide how much each direction contributes per level.
**Disadvantages:** two passes add latency over a single top-down FPN (~1.4x neck cost); global
attention on P5 still costs O(N^2) in tokens, noticeable above 1280px inputs.
**Performance notes:** at 640px, P5 is 20x20 = 400 tokens -- the attention is trivially cheap; the
upsample/concat traffic dominates, which is why gates use scalars, not per-pixel maps.

### 2.3 Head: ApexHead (anchor-free, quality-aware, NMS-based)

**What:** shared classification and regression towers (grouped-norm convs) applied to every
pyramid level; per location it predicts class logits, ltrb distances to box edges (softplus,
scaled by a learnable per-level scale and the stride), and a quality logit trained against the
IoU between the predicted box and its assigned ground truth. Inference score =
sqrt(sigmoid(cls) * sigmoid(quality)); decoding keeps top candidates per level and runs LOFOP's
native C++ class-aware NMS.

**Why anchor-free:** anchors add hyperparameters (sizes, ratios, per-dataset retuning) that
enterprise users should not have to own; point-based ltrb regression covers tiny-to-huge objects
through the pyramid itself.
**Why NMS instead of RT-DETR's NMS-free decoding:** one-to-one matching removes NMS but couples
recall to a fixed query budget (small dense scenes can exhaust queries), converges slower, and
produces export graphs (top-k gather chains) that are harder to run on edge runtimes. A dense
head + our 200x-accelerated native NMS keeps export trivially portable and recall unlimited. This
is the biggest deliberate divergence from the baseline.
**Advantages:** simple, robust, exportable; quality branch closes the cls/loc mismatch that
plagues dense heads.
**Disadvantages:** NMS is a hyperparameter (IoU threshold) and a latency term that query-based
models do not pay; crowded same-class scenes are the known weakness.
**Performance notes:** towers are shared across levels (parameter- and cache-friendly); decode
cost is dominated by NMS, which is exactly the op we moved to C++.

### 2.4 Label assignment: DynamicTopKAssigner

**What:** for each ground-truth box, candidate locations are points inside the box that also lie
within a center-prior radius (2.5 strides of the box center). Candidates are ranked by a cost that
combines classification score and IoU of the *current* predictions; each GT takes its top-k
candidates, where k is derived per-GT from the sum of its top IoUs (few good candidates -> small
k). Conflicts resolve to the lowest-cost GT.

**Why it exists:** fixed geometric assignment (e.g. "center 3x3") wastes supervision on poorly
matching points and starves small objects. Prediction-aware dynamic assignment (the concept behind
SimOTA and RT-DETR's matching alike) lets the model's own quality decide which points train as
positives, which is worth 1-2 mAP on small objects in published ablations.
**Advantages:** adapts to object size automatically; no per-dataset anchor tuning.
**Disadvantages:** assignment depends on predictions, so early training is noisier; the center
prior is a hyperparameter (2.5 strides) we inherit as a convention.
**Performance notes:** cost matrices are (num_gt x num_points-in-prior), computed with no_grad;
negligible next to the forward pass.

### 2.5 Losses

- **Classification:** sigmoid focal loss (gamma=2, alpha=0.25) over all locations -- dense heads
  are extremely class-imbalanced and focal weighting is the standard, well-understood fix.
- **Box:** GIoU loss on positives -- scale-invariant (small boxes are not out-shouted by big
  ones) and supplies gradients even for non-overlapping predictions.
- **Quality:** BCE against the IoU of the predicted box with its assigned GT.
- Total: `cls + 2.0 * giou + quality`, normalized by positive count (batch-averaged).

## 3. Model family and config

Sizes are pure config -- same code, different widths/depths:

| Variant | Backbone widths | Depths | Neck width | Params (measured) |
|---|---|---|---|---|
| `lofop-detect-n` | 32/64/128/256 | 1/1/2/1 | 64 | see `benchmarks/bench_detect.py` |
| `lofop-detect-s` | 48/96/192/384 | 1/2/4/2 | 96 | see `benchmarks/bench_detect.py` |

Configs live in `configs/lofop-detect/` and build through the registry:
`HUB.build(cfg.model)` returns a ready `LofopDetect`.

## 4. RT-DETR as an optional model

The registry makes the baseline a plug-in, not a fork: an `rtdetr` plugin can register
`model/RTDETR` (e.g. wrapping a third-party implementation with its own license and weights) and
every LOFOP tool -- configs, CLI, future trainer -- works with it unchanged. LOFOP itself ships no
RT-DETR code; the comparison harness treats it as an external reference.

## 5. Benchmark protocol vs RT-DETR

What we can measure now (CPU container, no training): parameter counts and forward latency via
`benchmarks/bench_detect.py`, against RT-DETR's published parameter counts as context. What the
real comparison requires (Phase 4): COCO train2017 training with EMA + cosine schedule, then
val2017 mAP / mAP-small / latency on identical GPU hardware, RT-DETR-R18 vs `lofop-detect-s`.
Until that run exists, no accuracy claims are made -- the protocol is committed so results are
reproducible, not asserted.

## 6. Improvement backlog (step 5 of the loop)

Ordered by expected return: distribution-based box regression (finer small-object localization),
denoising-style auxiliary supervision adapted to dense heads, backbone pretraining, quality-aware
NMS (Soft-NMS variant in the C++ kernel), INT8-calibration-friendly activation clipping.
