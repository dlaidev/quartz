---
title: "From Tensor Equations to Fast Kernels"
subtitle: "Neural Circuit Diagrams, equality saturation, and the missing compiler layer"
description: "A worked guide to axis-aware tensor semantics, streaming attention, and a possible pyncd–EggEvolve compiler pipeline."
date: 2026-09-13
draft: false
tags:
  - compilers
  - inference
  - gpu
  - equality-saturation
---

_Neural Circuit Diagrams, equality saturation, and the missing compiler layer_

An attention equation describes a function. A fast attention kernel describes a way to compute it: which values stay in registers, which tiles share memory, and which intermediate tensors never exist.

The difficult part is connecting those descriptions without losing either mathematical meaning or implementation freedom.

Neural Circuit Diagrams offer a language for the first half. Equality saturation offers a way to explore the second. The interesting possibility is a compiler that searches over algorithms, not just the tile sizes of an algorithm someone already wrote.

This guide develops that possibility through [pyncd](https://github.com/mit-zardini-lab/pyncd) and a proposed integration with EggEvolve, the kernel-search project used as the design target here. The integration is a proposal: the semantic export, algorithm-level rewrite rules, and GPU lowering described below still need to be connected.

## What you will learn

By the end, you should be able to explain:

- Why explicit index maps reveal more than tensor shapes alone.
- How broadcast axes differ from the axes an operation consumes.
- Why attention can be streamed without materializing its score matrix.
- What an e-graph contributes beyond an ordinary rewrite pass or autotuner.
- Which parts of a pyncd–EggEvolve compiler are available, and which must be built.

You need familiarity with tensors, softmax, and GPU memory hierarchy. No category theory background is assumed.

## 1. Start with a transformation you can check

Take a matrix $X$ and a pure, deterministic elementwise function $f$, such as squaring.

There are two ways to compute the same result:

$$
Y = \operatorname{transpose}(\operatorname{map}(f,X))
  = \operatorname{map}(f,\operatorname{transpose}(X)).
$$

At output coordinate $(i,j)$, both sides produce $f(X[j,i])$. The equality follows from indexing; we did not need a special property of squaring.

Let $\eta(i,j)=(j,i)$. Define reindexing by

$$
(\eta^*X)[p]=X[\eta(p)].
$$

The map runs **from output coordinates back to input coordinates**. This is the same direction used when a kernel computes the source address for an output element.

Our transpose example is an instance of a general law:

$$
\operatorname{map}_P(f,\eta^*X)
=
\eta^*\operatorname{map}_Q(f,X),
\qquad \eta:P\to Q.
$$

Here $P$ and $Q$ are index sets. Reindexing and pointwise computation commute.

This is a small but useful compiler theorem. It can move a transpose through an activation so a later pass can absorb the transpose into a consumer's indexing. It does not guarantee a speedup: the backend must decide whether the rearrangement is a view, fused address calculation, or actual data movement.

The word _deterministic_ matters. Copying one dropout result gives two copies of the same random mask. Copying the input and running two independent dropouts gives two different masks. Moving a computation across a reindexing that duplicates values can change correlations. A rewrite law needs its assumptions, not just its picture.

The categorical formulation makes laws like this compositional. A type describes an input or output space; a morphism describes a computation; composition connects them. A local equality can be substituted into a larger well-typed expression. The value is disciplined rewriting, not unfamiliar terminology. [Weaves, Wires, and Morphisms](https://arxiv.org/abs/2604.07242) develops the array-broadcasted formulation behind pyncd.

## 2. An axis has a role, not just a length

Consider softmax on a tensor $X[B,S,H]$:

$$
Y[b,:,h]=\operatorname{softmax}(X[b,:,h]).
$$

The primitive is a vector function $f:\mathbb{R}^{S}\to\mathbb{R}^{S}$. It runs independently for every $(b,h)$.

| Axis | Role for this softmax | Consequence                                     |
| ---- | --------------------- | ----------------------------------------------- |
| $S$  | Target axis           | Entries participate in the same normalization.  |
| $B$  | Broadcast axis        | Different batches are independent applications. |
| $H$  | Broadcast axis        | Different heads are independent applications.   |

A _weave_ records how the broadcast and target axes are interleaved in the tensor's axis order. Here that order is broadcast–target–broadcast. It says which subtensor one application of $f$ receives. It is not, by itself, a GPU lane layout or an LDS swizzle.

An important naming trap: pyncd uses a `TILED` marker for broadcast-degree axes. That marker does **not** mean a GPU tile size or a workgroup assignment has already been selected.

For an operation with several inputs, each input has its own access map. A simplified description is

$$
y[p]=f\bigl(x_1[\eta_1(p)],\ldots,x_k[\eta_k(p)]\bigr).
$$

The indexed values can themselves be target subtensors, not just scalars. The access maps describe reuse; the primitive describes computation within those subtensors.

This also separates a tuple of tensors from the axes within one tensor. Passing Q and K as two inputs is not the same operation as concatenating them along an axis. Keeping that distinction explicit is part of what makes [Neural Circuit Diagrams](https://arxiv.org/abs/2402.05424) more precise than an ordinary boxes-and-arrows architecture picture.

## 3. Read attention as index flow

Ignore batch and masking briefly. Let

$$
Q[q,h,d],\qquad K[x,h,d],\qquad V[x,h,d_v].
$$

With feature dimension $D_k$, the score operation is

$$
S[h,q,x]=\frac{1}{\sqrt{D_k}}\sum_{d=0}^{D_k-1} Q[q,h,d]K[x,h,d].
$$

Its primitive is a dot product over $d$. Its broadcast coordinate is $(h,q,x)$. The input access maps are

$$
\eta_Q(h,q,x)=(q,h),\qquad
\eta_K(h,q,x)=(x,h).
$$

Notice what is absent. Q's map ignores $x$; K's map ignores $q$. That exposes opportunities to reuse a Q tile across keys and a K tile across queries.

![Index flow exposes reuse. Each score selects one query vector and one key vector. The dot product acts on d; q and x enumerate independent scores.](images/tensor-kernels/axis-map.svg)

_Index flow exposes reuse. Each score selects one query vector and one key vector. The dot product acts on d; q and x enumerate independent scores._

Attention then normalizes along $x$ and contracts with V:

$$
P[h,q,:]=\operatorname{softmax}(S[h,q,:]),
\qquad
O[q,h,d_v]=\sum_x P[h,q,x]V[x,h,d_v].
$$

Query rows are independent. The key axis participates in normalization and reduction. Those are different scheduling opportunities: distribute work across query tiles, and stream key/value tiles through each query tile's state.

![Same attention, different storage. The materialized algorithm writes score/probability matrices. The streamed algorithm keeps only a query tile, a key/value block, and online state. Batch/head axes and score scaling are omitted. This is logical dataflow, not a measured GPU timeline.](images/tensor-kernels/attention-schedules.svg)

_Same attention, different storage. The materialized algorithm writes score/probability matrices. The streamed algorithm keeps only a query tile, a key/value block, and online state. Batch/head axes and score scaling are omitted. This is logical dataflow, not a measured GPU timeline._

That distinction is central to [FlashAttention on a Napkin](https://arxiv.org/abs/2412.03317). Its diagrammatic method exposes high-level tiling, streaming, and fusion strategies. It does not make the online-softmax identity, reduction semantics, or low-level hardware constraints unnecessary. Those must still be supplied and respected.

## 4. The streaming step, with actual numbers

For one query, attention is a normalized weighted sum:

$$
O=\frac{\sum_i e^{s_i}v_i}{\sum_i e^{s_i}}.
$$

Instead of retaining all scores, maintain a summary of the scores already seen:

$$
m=\max_i s_i,\qquad
\ell=\sum_i e^{s_i-m},\qquad
o=\sum_i e^{s_i-m}v_i.
$$

The output is $o/\ell$. In a real attention kernel, $o$ is a vector; the example below uses scalar values so every step is visible.

When a new block $J$ arrives, update:

$$
\begin{aligned}
m' &= \max\left(m,\max_{j\in J}s_j\right),\\
\alpha &= e^{m-m'},\\
\ell' &= \alpha\ell+\sum_{j\in J}e^{s_j-m'},\\
o' &= \alpha o+\sum_{j\in J}e^{s_j-m'}v_j.
\end{aligned}
$$

The rescaling is the entire trick. Old and new contributions must use the same reference maximum before being added.

Use scores $[0,1,2,3]$, values $[1,2,4,8]$, and blocks of two.

**First block.** Scores $[0,1]$ have maximum $m=1$:

$$
\ell=e^{-1}+1=1.367879,
\qquad
o=e^{-1}\cdot1+1\cdot2=2.367879.
$$

**Second block.** Scores $[2,3]$ raise the maximum to $m'=3$. Multiply the old summary by $\alpha=e^{-2}=0.135335$:

$$
\ell'=e^{-2}(e^{-1}+1)+(e^{-1}+1)=1.553002,
$$

$$
o'=e^{-2}(e^{-1}+2)+(4e^{-1}+8)=9.791975.
$$

| Scores processed | Maximum $m$ | Normalizer $\ell$ | Numerator $o$ | Current $o/\ell$ |
| ---------------- | ----------: | ----------------: | ------------: | ---------------: |
| $[0,1]$          |           1 |          1.367879 |      2.367879 |         1.731059 |
| $[0,1,2,3]$      |           3 |          1.553002 |      9.791975 |         6.305193 |

The direct result agrees:

$$
\frac{1+2e+4e^2+8e^3}{1+e+e^2+e^3}
\approx 6.305193.
$$

> **Interactive companion:** <a href="static/tensor-kernels/#online-softmax-demo" data-router-ignore>Step through the online-softmax recurrence</a>. The numerical table above contains the same complete worked example.

We retained a sufficient summary, not the previous block's normalized output. Averaging the two blocks' softmax outputs would be wrong because the blocks carry different total weights.

For GPU attention, the corresponding algorithm loads a K/V block, computes a score tile, updates the summary, and discards the score tile. The full quadratic score/probability intermediates need not be written to HBM. This reduces intermediate traffic and storage, not the quadratic arithmetic of dense attention.

The equations above describe real-number equality. For masked attention, exclude masked logits before computing each block's maximum and exponential contributions. Implementations also need rounding rules and defined behavior for empty or fully masked rows; blindly evaluating $-\infty-(-\infty)$ is not a valid initialization strategy.

## 5. Why an e-graph belongs here

A conventional rewrite pass chooses a transformation and proceeds with the modified program. That creates a phase-ordering problem: a locally unattractive rewrite may enable a valuable later one.

Over exact real arithmetic, consider

$$
a(b+c)-ab.
$$

Distributing multiplication temporarily makes the expression larger:

$$
ab+ac-ab.
$$

But it exposes cancellation, leaving $ac$. A rule that always rejects growth can miss this path.

An e-graph represents many equivalent expressions through shared structure. Equality saturation repeatedly applies trusted equalities while retaining alternatives. Extraction then chooses a representative according to a cost objective. An e-class means “these terms are equal under the supplied theory,” not “the system independently proved arbitrary code correct.” The [egg paper](https://arxiv.org/abs/2004.03082) explains the data structure, rebuilding, and analyses that make this practical.

For attention, alternatives might include materialized, tiled-unfused, and fused-streaming algorithms. Their equivalence needs a rule with an accumulator invariant, not merely an assertion that their outputs look similar on a few inputs.

An e-graph does not eliminate search limits. Rewrites can create enormous spaces, so saturation is usually bounded and guided. GPU extraction is also not simply “fewest operations”: fusion can lower HBM traffic while increasing live registers enough to cause spills. A useful extractor keeps several feasible candidates for compilation and measurement.

## 6. The missing layer between pyncd and EggEvolve

The proposed division of responsibility is straightforward: pyncd supplies a structured description of meaning; EggEvolve would provide the search and candidate-evaluation side. Connecting them requires more than serializing a graph.

It helps to separate three levels:

| Level     | Question answered                | Representative constructs                               |
| --------- | -------------------------------- | ------------------------------------------------------- |
| MathIR    | What function must be preserved? | Broadcast, reindex, contraction, normalization          |
| AlgIR     | Which algorithm computes it?     | Materialize, tile, fuse, stream, recompute              |
| BackendIR | How does the GPU execute it?     | Waves, lanes, LDS layout, instructions, pipeline stages |

These are proposed integration boundaries, not three existing repository APIs.

A pyncd export can seed MathIR with canonical axes, access maps, primitives, and shared dataflow. The new AlgIR must express implementation choices with enough detail to state their legality. A streaming rewrite, for example, should require a registered state-update law and preserve masks and output indexing.

![The proposed bridge. pyncd supplies a mathematical description. A new typed algorithm IR would connect trusted rewrites to extraction, lowering, correctness tests, and measured search. The arrows describe a proposal, not an existing integration.](images/tensor-kernels/compiler-loop.svg)

_The proposed bridge. pyncd supplies a mathematical description. A new typed algorithm IR would connect trusted rewrites to extraction, lowering, correctness tests, and measured search. The arrows describe a proposal, not an existing integration._

The loop would extract candidates, reject resource-infeasible schedules, lower survivors, run differential tests, and benchmark them. Measurements can calibrate the cost model. They cannot turn an invalid equality into a valid one.

For an initial backend, a few structured Triton templates are reasonable. A broader route is to lower toward MLIR: Linalg already exposes iteration spaces and affine indexing maps and supports tiling, fusion, promotion, and further lowering. pyncd would complement that infrastructure rather than replace it. [MLIR Linalg documentation](https://mlir.llvm.org/docs/Dialects/Linalg/) is a useful comparison point.

### Grouped-query attention exposes a useful reuse pattern

In grouped-query attention (GQA), several query heads share one key/value head. Suppose $H_q=G H_{kv}$. With contiguous head grouping, $h_q=h_{kv}G+g$. Factor the query-head axis as $(h_{kv},g)$:

$$
Q[b,q,h_{kv},g,d],\quad
K[b,x,h_{kv},d],\quad
V[b,x,h_{kv},d_v].
$$

The K/V access maps omit $g$. Instead of hiding head sharing behind division in an address expression, the representation makes it a projection: all query heads in a group access the same K/V head.

A different head ordering requires an explicit permutation. Factoring describes semantic sharing; the backend still has to realize physical reuse.

That does not guarantee the best schedule processes all $G$ heads together. It makes the tradeoff available: more K/V reuse versus more live query/output state. The backend search decides how much sharing fits the target's resources.

An LLM could propose tile choices, rewrite schedules, or algorithm sketches. It should not grant itself permission to merge expressions. A useful boundary is: **search proposes; registered semantics constrains; hardware measures.**

## 7. Decide what “equivalent” means before searching

Real-number algebra, floating-point execution, and model-quality preservation are different contracts.

Reassociating a sum is valid over reals but can change floating-point results. Quantization changes represented values. Replacing softmax with a different normalization changes the model function. None belongs silently inside an e-class claiming bitwise equality.

Likewise, a low-level rewrite needs the whole transformation. Padding GEMM dimensions requires correct masked loads and output cropping. Changing an LDS stride requires updating allocation and producer/consumer address maps consistently. Successful compilation proves neither of those conditions.

For a first prototype, choose a narrow contract: fixed shapes and dtypes, deterministic inference, explicit masks, and documented absolute/relative output tolerances. Test awkward dimensions, extreme logits, and masked rows—not just random square tensors.

Tolerance-based closeness is not transitive: A can be close to B, and B close to C, while A is not close to C. It therefore cannot justify ordinary e-class merging. Keep approximate candidates separate and compare each against a fixed reference or maintain explicit error bounds.

A proof of a high-level rewrite, implementation tests, and a compiler backend each cover different failure modes. Calling the whole pipeline “proof-guided” should not obscure the trusted code and unverified lowering beneath it.

## 8. A small milestone that would demonstrate the idea

Start with one forward GQA operation on one AMD target. Export its pyncd meaning into a typed IR. Add a small set of rules for reindexing, tiling, fusion, and one online-softmax accumulator. Produce materialized and streamed candidates, then compile, compare, and time them.

The important result is not a promised speedup. It is an inspectable path from the same mathematical input to genuinely different algorithms, with an explicit correctness contract and real measurements.

Only after that path works should the project expand to more operators, hardware families, or LLM-guided exploration. Otherwise a sophisticated search loop can spend its entire budget optimizing assumptions that were never implemented.

## Check your understanding

Try answering before opening the hints.

<details>
<summary>What would break if softmax were moved across an arbitrary reshape as though it were elementwise?</summary>

The reshape might change which elements belong to one normalization group. The map/reindex law applies to the chosen local primitive and its broadcast structure; it does not license changing that primitive's target grouping.

</details>

<details>
<summary>Why must the first block's state be rescaled when a larger score arrives?</summary>

Its exponentials were measured relative to the old maximum. Multiplying both numerator and denominator contributions by the same factor converts them to the new reference maximum without changing their ratio.

</details>

<details>
<summary>What does factoring the GQA head axis reveal, and what does it leave undecided?</summary>

It reveals that K/V accesses are invariant over the group coordinate. It leaves physical data placement, wave mapping, and the profitable amount of simultaneous head processing undecided.

</details>

<details>
<summary>Why can fewer HBM bytes still produce a slower extracted kernel?</summary>

The schedule may increase register pressure, spills, synchronization, redundant computation, or occupancy loss. These costs interact, so a byte-count objective alone is insufficient.

</details>

## Eight terms worth keeping

| Term                | Meaning here                                                      |
| ------------------- | ----------------------------------------------------------------- |
| Morphism            | A typed computation that can compose with other computations.     |
| Reindexing          | Selecting input coordinates through an output-to-input map.       |
| Broadcast degree    | The index space over independent applications of a primitive.     |
| Weave               | The interleaving of broadcast axes and primitive-target axes.     |
| Denotation          | The mathematical meaning of an expression or algorithm.           |
| E-class             | Expressions treated as equivalent under registered rules.         |
| Equality saturation | Applying equalities while retaining alternative expressions.      |
| Extraction          | Selecting an implementable representative using a cost objective. |

## Implementation scope

At [pyncd `13c1de0`](https://github.com/mit-zardini-lab/pyncd/tree/13c1de0296120e07b959d4de23e4d1151abeb8d8), the documented pieces include algebraic terms, symbolic axes, graph conversion, serialization, rendering support, and conversion to PyTorch modules. This is the semantic foundation for the proposal; the algorithm-search layer and Triton/MLIR lowering described here are additional work.

The pyncd–EggEvolve bridge in this article is a design proposal, with no end-to-end implementation or benchmark result claimed.

## Reading, in a useful order

1. [Neural Circuit Diagrams](https://arxiv.org/abs/2402.05424) — Learn the visual distinction between tensor axes, separate values, and broadcast operations.
2. [Weaves, Wires, and Morphisms](https://arxiv.org/abs/2604.07242) — Read the formal account of index maps, weaves, and compositional model terms.
3. [FlashAttention on a Napkin](https://arxiv.org/abs/2412.03317) — Follow the move from architecture diagrams to IO-aware tiling and streaming. The accompanying [GPU MODE talk](https://www.youtube.com/watch?v=hAoY2bpRIKg) presents the diagrammatic approach.youtube.com/watch?v=hAoY2bpRIKg&t=2187s), then [kernel composition at 51:11](https://www.youtube.com/watch?v=hAoY2bpRIKg&t=3071s).
4. [egg: Fast and Extensible Equality Saturation](https://arxiv.org/abs/2004.03082) — Understand what e-graphs retain, how equalities are applied, and where domain analyses enter.
5. [MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/) — Compare these semantic ideas with existing structured-compiler machinery.
6. [pyncd source and examples](https://github.com/mit-zardini-lab/pyncd) — Connect the mathematical language to concrete term construction and execution, while keeping implementation scope in view.

## Related posts

- [Triton Linear Layouts](2025-06-22-linear-layouts.md) — mapping logical tensor coordinates to GPU execution.
- [CuTe Basics](2025-05-10-cute-basics.md) — layouts, tensors, and composition.

_Diagrams are original explanatory schematics. The worked example illustrates the arithmetic; it is not a GPU benchmark._
