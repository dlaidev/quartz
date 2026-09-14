---
title: "Triton Value Dataflow"
date: 2025-05-26
tags:
  - PyTorch
  - AI
  - Compilers
  - Backend
  - TorchInductor
  - Parallel-Computing
  - Performance-Optimization
  - AI-Inference
  - Value-Dataflow-Analysis
  - Triton
draft: false
---

A compiler needs different facts to remove an unused computation, move a load, or fuse two kernels. SSA use lists describe value dependencies. Memory effects and alias analysis constrain accesses to storage. Dataflow analysis propagates facts through branches and loops until those facts stop changing.

This post develops reaching definitions and liveness with a CPU model, then maps them to MLIR and Triton. The Python model runs; the MLIR integration is a source guide.

## Recorded fusion experiment

The original H100 experiment compared two matrix-multiplication kernels with a fused implementation:

| Version | Kernel time (µs) | Speedup |
| --- | ---: | ---: |
| 2‑kernel (spills) | 470 | 1× |
| Fused | 4.3 | 110× |

These are the original recorded values, including the rounded speedup. This revision did not rerun the experiment.

| Schedule | Intermediate `%x` |
| --- | --- |
| Separate producer and consumer | Write `%x` to device memory, then reload it |
| Fused producer and consumer | Consume partial results while they remain on-chip |

The timing difference motivates dependency analysis. It does not establish that memory traffic alone caused the entire speedup.

## SSA values and memory dependencies

In MLIR, a value is an operation result or a block argument. An SSA value has one definition. A block argument receives a value from the incoming control-flow edge; MLIR uses block arguments for the role commonly served by phi nodes.[2]

Consider this pseudocode:

```text
a = load A
store A, c
use a
```

The load produces a value. The later store changes the contents of `A`; it does not change the value already held in `a`. Moving the store above `use a` is therefore not, by itself, a write-after-read violation. Moving it above the **load** can change what the load observes.

Memory dependencies concern potentially overlapping accesses:

| Dependency              | Original order           | Unsafe change without further proof               |
| ----------------------- | ------------------------ | ------------------------------------------------- |
| Read after write (RAW)  | `store A; load A`        | Load before the store                             |
| Write after read (WAR)  | `load A; store A`        | Store before the load                             |
| Write after write (WAW) | `store A, x; store A, y` | Reverse the stores when the final contents matter |

These examples assume ordinary sequential accesses to the same location. With different pointers, an alias analysis must establish whether their byte ranges overlap. Atomics, barriers, asynchronous operations, and accesses from other threads add ordering requirements. A set of SSA definitions cannot establish those requirements.

SSA also changes what reaching definitions means. For an operand `%x`, its defining operation is already explicit. Collecting every SSA result seen earlier in a function adds little useful information and can include values outside the operand's scope. Useful value analyses instead ask whether `%x` is constant, divisible by an alignment, or known to have contiguous elements. Triton's `AxisInfoAnalysis` is an example of the latter approach.[3]

## Reaching definitions on mutable variables

Classical reaching definitions operates on assignments to variables that can be reassigned. A definition reaches a point if some control-flow path carries it there without an intervening assignment to the same variable. This is a **may** analysis: membership means possible, rather than guaranteed.

Use assignment labels to distinguish definitions:

```text
entry: x0: x = 0; i0: i = 0
       branch left or right
left:  x1: x = 1; goto join
right: x2: x = 2; goto join
join:  y0: y = x; goto loop
loop:  x3: x = x + i; i1: i = i + 1
       branch loop or exit
exit:  use(x, y)
dead:  x_dead: x = 99; goto join   # no path from entry
```

At `join`, either `x1` or `x2` may supply `x`. At the next iteration of `loop`, `x3` may supply it. The unreachable block must not contribute `x_dead`, even though it has an edge to `join`.

Let $D$ be the finite set of definition labels in reachable blocks. For block $B$:

- $GEN_B$ contains the last definition of each variable assigned within the block.
- $KILL_B$ contains definitions of variables assigned within the block.
- $IN_B$ and $OUT_B$ describe the state immediately before and after the block.

Our implementation includes the generated definitions in `KILL`; the union with `GEN` restores them. The equations are:

$$
IN_B = \bigcup_{P \in pred(B)} OUT_P
$$

$$
OUT_B = GEN_B \cup (IN_B \setminus KILL_B)
$$

Initialize every state to the empty set. The entry boundary is empty because this example has no incoming definitions. A model of function parameters or externally initialized variables would need explicit boundary definitions.

The state space is the powerset lattice $\mathcal{P}(D)$ ordered by inclusion. Bottom is the empty set, top is $D$, and the least upper bound, or join, is union. The transfer function is monotone: adding an incoming definition cannot remove an outgoing definition. Although `KILL` removes facts while transferring across a block, it does not cause successive solver states to shrink when iteration starts at bottom.

A worklist revisits successors when a forward state changes. Loops require revisits because information can return along a backedge. Finite lattice height and monotone updates guarantee termination here. Infinite domains, such as unbounded intervals, may need widening or another convergence rule. The fixed point conservatively includes paths through both sides of every branch; it does not prove that a particular path is feasible.

## Liveness runs backward

A variable is live before an instruction if a later instruction may read its current value before another assignment replaces it. Define $USE_B$ as variables read before their first definition within $B$, and $DEF_B$ as variables assigned in $B$.

$$
LIVEOUT_B = \bigcup_{S \in succ(B)} LIVEIN_S
$$

$$
LIVEIN_B = USE_B \cup (LIVEOUT_B \setminus DEF_B)
$$

The join uses **successors**. When a live-in state changes, predecessors need revisiting. The exit boundary is empty unless the surrounding environment requires particular variables after the function returns.

Instruction order matters when building `USE`. In `x = x + i`, the right-hand `x` is a use of the old value. It remains upward-exposed even though the instruction also defines `x`. In `x = 1; use(x)`, the use does not make `x` live at block entry.

Reaching definitions tracks assignment identities; this liveness analysis tracks variable names. In SSA, liveness tracks individual SSA values and must handle block arguments and their incoming operands on the appropriate edges. Joining every incoming operand into every predecessor would invent live ranges on paths that never use those operands.

## Run the CPU example

Download [dataflow.py](code/triton-analysis/dataflow.py), [linear_layout.py](code/triton-analysis/linear_layout.py), and [test_analysis.py](code/triton-analysis/test_analysis.py) into one directory. The tests cover both this post and [Triton Linear Layouts](2025-06-22-linear-layouts.md). Python 3 and its standard library are sufficient.

From the repository root:

```bash
uv run --no-project python -B content/code/triton-analysis/dataflow.py
uv run --no-project python -B -m unittest discover -s content/code/triton-analysis -v
```

The forward update in the companion is:

```python
inside = set().union(*(outs[p] for p in pred[b]))
out = gen[b] | (inside - kill[b])
```

The backward update is:

```python
out = set().union(*(ins[s] for s in edges[b] if s in active))
inside = use[b] | (out - defs[b])
```

`active` comes from a reachability traversal starting at `entry`. The solver puts every active block on the initial worklist. A change adds dependent blocks back to the worklist without adding duplicate pending entries.

Selected output from the executed program:

```text
join: RD_IN={i0, x1, x2} RD_OUT={i0, x1, x2, y0} LIVE_IN={i, x} LIVE_OUT={i, x, y}
loop: RD_IN={i0, i1, x1, x2, x3, y0} RD_OUT={i1, x3, y0} LIVE_IN={i, x, y} LIVE_OUT={i, x, y}
exit: RD_IN={i1, x3, y0} RD_OUT={i1, x3, y0} LIVE_IN={x, y} LIVE_OUT={}
dead: unreachable
```

`x1`, `x2`, and `x3` reach the loop entry through different paths. Only `x3` reaches its exit because the loop body always assigns `x`. `y` stays live through the loop despite having no use inside it: `exit` reads it.

The combined suite passed 11 tests. Dataflow tests check exact join and loop states, unreachable-block exclusion, instruction ordering, and independence from initial worklist order. They also recompute each final transfer instruction by instruction and check every fixed-point equation. This is a finite CFG model with explicit uses and definitions, without an MLIR parser, alias analysis, or GPU execution.

## Dense and sparse analysis in MLIR

Dense analysis attaches a state to program points, such as before and after an operation. It suits facts about the surrounding program state. Sparse analysis attaches facts to SSA values and propagates them through value dependencies. These names describe where the analysis stores and propagates state; they do not prescribe a dense bit vector or a sparse set container.[9][10]

| Analysis question                                       | Useful state placement                        |
| ------------------------------------------------------- | --------------------------------------------- |
| Which mutable-variable definitions reach this point?    | Dense state at program points                 |
| What constant or alignment is known for this SSA value? | Sparse lattice on values                      |
| Which variables may be used after this block?           | Backward state at CFG boundaries in our model |

Constant propagation often uses a domain with bottom, individual constants, and an overdefined state. Joining the same constant retains it; joining different constants yields overdefined. Bottom, an empty set of reaching definitions, and an unreachable block are not interchangeable concepts. Their meanings depend on the analysis domain and execution model. MLIR's tutorial introduces lattices, but its illustrated `ForwardDataFlowAnalysis` API differs from the source revision used below.[1][8]

## Integration with Triton: source guide, not a plugin

The source references use Triton **v3.3.1**, commit `d654e0f2d91f07496454e0fcbec2a9b97df37d47`. Its `cmake/llvm-hash.txt` selects LLVM commit `a66376b0dc3b2ea8a84fda26faca287980986f78`; the MLIR header citations below use that exact revision.[4] This avoids combining an arbitrary LLVM release with Triton's expected API.

In those headers:

- `mlir::DataFlowSolver` owns analyses and their states. `load<AnalysisT>()` constructs an analysis; `initializeAndRun(Operation *)` initializes it and runs to a fixed point.[8]
- `mlir::dataflow::DenseForwardDataFlowAnalysis<LatticeT>` operates on a subclass of `AbstractDenseLattice`. Its operation hook receives the state before an operation and a mutable state after it.[9]
- `mlir::dataflow::SparseForwardDataFlowAnalysis<StateT>` operates on sparse lattices. Its operation hook receives operand lattices and result lattices. `dataflow::Lattice<T>` supplies a typed sparse lattice wrapper.[10]

The dense operation hook at this revision has this signature. This is an API excerpt, not a complete implementation:

```cpp
LogicalResult visitOperation(Operation *op,
                             const LatticeT &before,
                             LatticeT *after) override;
```

A transfer implementation must report and propagate state changes through the framework. Updating a container without notifying dependents can leave downstream facts stale. Entry-state hooks must conservatively handle externally supplied values. Region branches, calls, and executable edges require the appropriate interfaces, transfer rules, and supporting analyses; recursively walking operations is insufficient to model their control flow.[8][9][10]

Triton's `AxisInfo.cpp` supplies a concrete integration reference. `AxisInfoAnalysis` derives from `SparseForwardDataFlowAnalysis<dataflow::Lattice<AxisInfo>>`. `ModuleAxisInfoAnalysis::initialize` obtains a solver through `createDataFlowSolver`, loads the analysis, and calls `initializeAndRun` on isolated operations during its walk. Read that implementation and its solver helper when adding a related analysis.[3]

For a custom pass, first choose the fact domain and the IR level where the needed information exists. Implement and test transfer rules against the checkout's headers. Then link the implementation into the intended tool, register the pass, and add it explicitly to the pipeline. Adding a CMake library alone does not make a command-line pass available. A version-specific out-of-tree plugin also needs compatible registration, linkage, and loading support. This post has not compiled or tested either integration path.

Use valid IR fixtures for diamonds, backedges, block arguments, unreachable regions, calls, and unknown operations. Print deterministic analysis results through a test pass and check them with FileCheck. Debug-only logging can go to a different stream and can depend on build configuration; it should not be the sole testing interface. Recompute or invalidate analysis states after transforming the IR.[8]

## What the facts permit

Liveness can expose a short-lived intermediate, but it does not promise that fusion will keep the intermediate in registers. A fused schedule may increase live storage, require shared-memory exchange, reduce occupancy, or change synchronization. Layout and lowering also determine how many physical registers a tensor value requires.

Dead-store elimination requires proof that no relevant read can observe the store before an overwrite or the end of the storage lifetime. Dead-value elimination requires checking operation effects as well as unused results. Fusion needs memory-dependence and synchronization reasoning in addition to SSA dependencies. The CPU analysis establishes its set equations; it makes no latency, register-allocation, or kernel-speedup claim.

## Sources

[1] https://mlir.llvm.org/docs/Tutorials/DataFlowAnalysis — MLIR dataflow tutorial

[2] https://mlir.llvm.org/docs/LangRef — MLIR language reference

[3] https://raw.githubusercontent.com/triton-lang/triton/d654e0f2d91f07496454e0fcbec2a9b97df37d47/lib/Analysis/AxisInfo.cpp — lib/Analysis/AxisInfo.cpp

[4] https://raw.githubusercontent.com/triton-lang/triton/d654e0f2d91f07496454e0fcbec2a9b97df37d47/cmake/llvm-hash.txt — cmake/llvm-hash.txt

[8] https://raw.githubusercontent.com/llvm/llvm-project/a66376b0dc3b2ea8a84fda26faca287980986f78/mlir/include/mlir/Analysis/DataFlowFramework.h — mlir/include/mlir/Analysis/DataFlowFramework.h

[9] https://raw.githubusercontent.com/llvm/llvm-project/a66376b0dc3b2ea8a84fda26faca287980986f78/mlir/include/mlir/Analysis/DataFlow/DenseAnalysis.h — mlir/include/mlir/Analysis/DataFlow/DenseAnalysis.h

[10] https://raw.githubusercontent.com/llvm/llvm-project/a66376b0dc3b2ea8a84fda26faca287980986f78/mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h — mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h
