---
title: "CuTe and CUTLASS Basics"
date: 2025-05-10
tags:
  - CUDA
  - GPU-Programming
  - Parallel-Computing
  - Performance-Optimization
  - AI-Inference
  - CUTLASS
  - CuTe
draft: false
---

CuTe describes how logical tensor coordinates map to storage offsets. CUTLASS uses those layouts to express tensor views and distribute work across threads. This tutorial covers the C++ layout interface, composition, and inverse coordinates. It ends with CPU checks and a procedure for testing memory access on a GPU.

This worklog began after NVIDIA announced Python support for CuTe/CUTLASS at GTC 2025. My intended target was an attention kernel on an RTX 4090. This entry covers the layout operations needed before tiling that kernel; its examples use the C++ interface.

The progression is concrete: construct two layouts, print their offsets, compose them, and check the result. Each cell in a printed table is a storage offset. The row and column labels are the coordinates supplied to the layout.

| Task | Question answered by its output |
| --- | --- |
| Row-major and column-major layouts | How do strides change the offset assigned to a coordinate? |
| Layouts A and B | What does each map do before composition? |
| Composed layout C | What happens when B's output becomes A's scalar input? |
| Coordinate recovery | How does a logical ordinal differ from a physical offset? |
| Profiling | How does a kernel assign these accesses to threads? |

## Shape, stride, and offset

For a flat two-dimensional layout with shape `(M,N)` and stride `(s0,s1)`, the coordinate `(r,c)` maps to the element offset

$$
L(r,c) = r s_0 + c s_1,
\qquad 0 \le r < M,\quad 0 \le c < N.
$$

CuTe writes this as `(M,N):(s0,s1)`. Shape bounds the coordinates; stride determines the index mapping.[6] An element address adds the base pointer and scales the offset by the element size. The layout's offset map is linear in its natural coordinates; the address map is affine because it includes the base address.

```cpp
using namespace cute;
auto row_major = make_layout(make_shape(4, 4), make_stride(4, 1));
auto col_major = make_layout(make_shape(4, 4), make_stride(1, 4));
```

Original `cute::print_layout()` output:

Row-major `(4,4):(4,1)`:

```text
       0    1    2    3
    +----+----+----+----+
 0  |  0 |  1 |  2 |  3 |
    +----+----+----+----+
 1  |  4 |  5 |  6 |  7 |
    +----+----+----+----+
 2  |  8 |  9 | 10 | 11 |
    +----+----+----+----+
 3  | 12 | 13 | 14 | 15 |
    +----+----+----+----+
```

Column-major `(4,4):(1,4)`:

```text
       0    1    2    3
    +----+----+----+----+
 0  |  0 |  4 |  8 | 12 |
    +----+----+----+----+
 1  |  1 |  5 |  9 | 13 |
    +----+----+----+----+
 2  |  2 |  6 | 10 | 14 |
    +----+----+----+----+
 3  |  3 |  7 | 11 | 15 |
    +----+----+----+----+
```

Both contain sixteen elements. Their storage order differs. CuTe's `print_layout(layout)` displays this coordinate-to-offset table; it does not read tensor values.[6]

Read each table by choosing a row and a column, then looking up the number at their intersection. In the row-major table, moving right advances the offset by one. In the column-major table, moving down advances it by one. The coordinate grid has not changed; the strides changed the storage order.

Strides need not describe compact storage. Shape `(6,2)` with stride `(8,2)` has twelve coordinates and offsets up to 42. A backing allocation indexed from zero therefore needs at least 43 elements. Shape size counts coordinates, while `cosize` describes the codomain extent for these nonnegative-stride layouts.[5] A zero stride can make several coordinates refer to the same element.

## A scalar argument is a logical ordinal

CuTe also accepts `layout(i)`. It expands `i` into coordinates in first-mode-fastest order before applying the strides. This expansion depends on the shape, independently of the layout's storage order.[6]

For shape `(4,3)`:

$$
r = i \bmod 4,\qquad c = \lfloor i/4 \rfloor.
$$

With stride `(3,1)`, `layout(7)` uses coordinate `(3,1)` and returns offset 10. It does not mean "read the element at physical offset 7."

For a hierarchical shape, CuTe applies the same coordinate decomposition within its nested modes. The final offset remains a sum of natural coordinates times strides. As a function of a scalar ordinal, that mapping can contain integer division and remainder operations.[6]

## Composition: apply B, then A

Use the original example:

```cpp
using namespace cute;
auto A = make_layout(make_shape(_6{}, _2{}), make_stride(_8{}, _2{}));
auto B = make_layout(make_shape(_4{}, _3{}), make_stride(_3{}, _1{}));
auto C = composition(A, B);
```

The `_N` types encode compile-time integers. Ordinary integers describe runtime values; static information lets CuTe simplify the resulting layout type.[6][7]

The original worklog constructed these layouts with runtime integers:

```cpp
auto A = make_layout(make_shape(6, 2), make_stride(8, 2));
auto B = make_layout(make_shape(4, 3), make_stride(3, 1));
auto C = composition(A, B);
```

### Layout A: `(6,2):(8,2)`

```text
       0    1
    +----+----+
 0  |  0 |  2 |
    +----+----+
 1  |  8 | 10 |
    +----+----+
 2  | 16 | 18 |
    +----+----+
 3  | 24 | 26 |
    +----+----+
 4  | 32 | 34 |
    +----+----+
 5  | 40 | 42 |
    +----+----+
```

A has six rows and two columns. Moving down adds eight to the offset; moving right adds two. The gaps are part of the layout. It does not pack its twelve coordinates into twelve consecutive storage locations.

### Layout B: `(4,3):(3,1)`

```text
       0    1    2
    +----+----+----+
 0  |  0 |  1 |  2 |
    +----+----+----+
 1  |  3 |  4 |  5 |
    +----+----+----+
 2  |  6 |  7 |  8 |
    +----+----+----+
 3  |  9 | 10 | 11 |
    +----+----+----+
```

B has four rows and three columns. Its row-major strides assign consecutive offsets across each row.

### Composed layout C

The original `print_layout` output was:

```text
((2,2),(3,1)):((24,2),(8,2))
       0    1    2
    +----+----+----+
 0  |  0 |  8 | 16 |
    +----+----+----+
 1  | 24 | 32 | 40 |
    +----+----+----+
 2  |  2 | 10 | 18 |
    +----+----+----+
 3  | 26 | 34 | 42 |
    +----+----+----+
```

C accepts B's coordinates. Its cells contain the offsets obtained after applying both layouts. The nested shape in the printed type describes how CuTe represents this composed mapping; it does not introduce another tensor allocation.

Composition means `C(i) = A(B(i))`. B's result becomes A's scalar logical input. This operation has shape and stride divisibility requirements; equal rank alone does not establish validity.[7] In this example, B produces every integer from 0 through 11, within A's twelve-element logical domain.

For a two-dimensional input `(r,c)`, calculate:

$$
\begin{aligned}
q &= 3r+c,\\
A(q) &= 8(q\bmod 6)+2\lfloor q/6\rfloor.
\end{aligned}
$$

The second line decomposes `q` according to A's shape `(6,2)`. The original printed C type was `((2,2),(3,1)):((24,2),(8,2))`. Its offsets are:

| r \\ c |   0 |   1 |   2 |
| ------ | --: | --: | --: |
| 0      |   0 |   8 |  16 |
| 1      |  24 |  32 |  40 |
| 2      |   2 |  10 |  18 |
| 3      |  26 |  34 |  42 |

For example, `B(2,1)` is 7. A expands 7 to `(1,1)`, then computes `8*1 + 2*1 = 10`.

The calculation has four steps:

| Step | Input | Output |
| --- | --- | --- |
| Read B's coordinate | Row 2, column 1 | `B(2,1) = 7` |
| Pass B's result to A | Scalar logical input `7` | `A(7)` |
| Expand A's scalar input using shape `(6,2)` | `7` | Coordinate `(1,1)` |
| Apply A's strides `(8,2)` | `(1,1)` | Offset `10` |

That is why C's row 2, column 1 contains 10. Do not search A's cells for the value 7: composition passes 7 as A's input, not as an offset to invert.

```text
B coordinate (2,1)
        |
        | B: apply strides (3,1)
        v
   scalar result 7
        |
        | A: expand ordinal using shape (6,2)
        v
A coordinate (1,1)
        |
        | A: apply strides (8,2)
        v
   storage offset 10
```

The row stride changes when `r` crosses the boundary between 1 and 2. A hierarchical layout can represent the result as `((2,2),3):((24,2),8)`. The first row coordinate splits into `(r % 2, r / 2)`. CuTe's printed type may retain size-one modes, particularly when inputs use runtime integers; compare mappings rather than type spelling.[7]

These two checks exercise different input forms:

```cpp
for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 3; ++c)
        assert(C(r, c) == A(B(r, c)));

for (int i = 0; i < size(B); ++i)
    assert(C(i) == A(B(i)));
```

The scalar loop visits B's coordinates in first-mode-fastest order. It does not visit the table row by row.

The original worklog checked every scalar input with:

```cpp
for (int i = 0; i < size(C); i++) {
    assert(C(i) == A(B(i)));
}
```

The composition assertions passed in the recorded worklog. The CPU companion below also checks the displayed offsets against explicit expected values.

## Inverting an offset

Three operations have different meanings:

| Operation             | Input                                        | Result                  |
| --------------------- | -------------------------------------------- | ----------------------- |
| `B(i)`                | Logical scalar ordinal                       | Storage offset          |
| `coalesce(B)(i)`      | Logical scalar ordinal                       | The same storage offset |
| `B.get_hier_coord(p)` | Storage offset in a supported compact layout | Natural coordinate      |

`coalesce` simplifies a layout while preserving its size and scalar-input mapping. It does not produce coordinates or invert the mapping.[7]

The earlier draft used `auto coord = coalesce(B)(i)` and described the result as a coordinate. That description was incorrect: the call still returns an offset. Keep that distinction when reading the two input conventions below.

For compact row-major B, physical offset `p` has inverse `(p / 3, p % 3)`. Thus physical offset 7 maps to `(2,1)`, although logical ordinal 7 maps to `(3,1)`.

```cpp
auto coord = B.get_hier_coord(7);
assert(get<0>(coord) == 2);
assert(get<1>(coord) == 1);
assert(B(coord) == 7);
```

CUTLASS v3.9.0 documents these inverse-coordinate helpers as valid for compact layouts. `get_flat_coord` returns a flat coordinate with the layout's top-level rank; `get_hier_coord` preserves the shape hierarchy.[5] The sparse A above falls outside that documented compact-layout precondition. Some offsets have no preimage. With aliased strides, an offset may have several preimages. For an arbitrary small layout, enumerate coordinates and collect every coordinate with the requested offset; reject missing or ambiguous results when the caller requires a unique inverse.

## Run the checks on a CPU

The companion [checks.py](/code/cuda-basics/checks.py) implements the flat shape/stride formulas without CUDA dependencies. It checks the composition table against explicit expected values, checks compact inverses, and distinguishes holes from aliases. Run from the repository root:

```sh
uv run --no-project python content/code/cuda-basics/checks.py
```

The layout portion printed this output during local verification:

```text
0 8 16
24 32 40
2 10 18
26 34 42
```

The program also checks the launch arithmetic used in [[2025-05-17-cuda-basics|CUDA threads and blocks]] and ends with `CPU reference checks passed`. These are integer-model checks, without GPU execution or timing.

[layouts.cpp](/code/cuda-basics/layouts.cpp) expresses the same checks with the CuTe API. It targets CUTLASS v3.9.0, commit `e94e888df3551224738bfa505787b515eae8352f`. With CUTLASS and CUDA Toolkit headers installed:

```sh
g++ -std=c++17 -Wall -Wextra \
  -I"$CUTLASS/include" -I"$CUDA_HOME/include" \
  content/code/cuda-basics/layouts.cpp -o /tmp/cute-layouts
/tmp/cute-layouts
```

This program contains host code and launches no kernels. The local check compiled it with GCC 13.3.0 and the pinned CUTLASS source. The executable printed the composition table above and `CuTe layout checks passed`.

The machine had no CUDA Toolkit installation. The check used isolated header packages instead:

```sh
uv pip install --target /tmp/cute-cuda-headers \
  nvidia-cuda-runtime-cu12==12.9.79 \
  nvidia-cuda-cccl-cu12==12.9.27 \
  nvidia-cuda-nvcc-cu12==12.9.86

g++ -std=c++17 -Wall -Wextra \
  -I"$CUTLASS/include" \
  -I/tmp/cute-cuda-headers/nvidia/cuda_runtime/include \
  -I/tmp/cute-cuda-headers/nvidia/cuda_nvcc/include \
  -I/tmp/cute-cuda-headers/nvidia/cuda_cccl/include \
  content/code/cuda-basics/layouts.cpp -o /tmp/cute-layouts
/tmp/cute-layouts
```

Set `CUTLASS` to the source checkout before either build command. The build emitted unused-parameter warnings from CUTLASS headers. It required no GPU or driver and verified the actual CuTe composition and inverse-coordinate APIs in addition to the independent Python arithmetic.

## Coalescing and divergence

Memory coalescing groups the addresses requested by active lanes of one warp memory instruction. Branch divergence occurs when lanes in a warp follow different control-flow paths.[3][4] A layout determines offsets; a kernel determines which lane uses each coordinate and which instructions it executes.

For row-major storage, assigning consecutive lanes to consecutive columns produces adjacent element addresses. Assigning them to rows produces addresses separated by the row stride. Both assignments can run the same instruction sequence without a branch. A noncontiguous address pattern alone is insufficient evidence of divergence.

Conversely, an `if (i < n)` guard can leave only part of a warp doing useful work while its remaining accesses are adjacent. Coalescing, useful-lane count, and occupancy measure different properties. CuTe's algebraic `coalesce` operation also differs from hardware memory coalescing: simplifying a layout expression does not establish how a kernel accesses memory.

![Illustrative warp mask with 20 active lanes and 12 inactive lanes](/images/warp_divergence_ncu.png)

| Original profiling note | Recorded value |
| --- | ---: |
| Average active threads per warp | 20.5 out of 32 |

The image shows one lane mask; the table preserves the reported average. This revision did not rerun the profile.

The original layout plots remain available below. Their labels describe a separate layout example, not the sparse A used in the composition calculation above.

![Original layout A plot](/images/layout_a.png)

![Original layout B plot](/images/layout_b.png)

![Original composed-layout plot](/images/layout_composed.png)

## Measure a kernel separately

A GPU experiment must supply a kernel that turns lane IDs into coordinates, performs observable loads or stores, and checks the output against a reference. Record the GPU, toolkit, kernel source, launch dimensions, allocation alignment, and element type. Keep problem size and work constant when comparing layouts.

Use a profiler to inspect global-memory sectors for the relevant load/store instructions, branch behavior, and occupancy separately. Resource usage and device limits constrain residency; additional resident warps do not guarantee a faster kernel.[4] Timing must specify whether allocation, initialization, transfers, and synchronization are included.

The checks above establish the coordinate mapping and its composition. A performance claim requires that additional kernel and measurement record.

## What the examples establish

| Operation | Meaning |
| --- | --- |
| `make_layout(shape, stride)` | Define the coordinate-to-offset map |
| `print_layout(L)` | Display offsets at logical coordinates |
| `L(i)` | Expand a scalar logical ordinal, then apply the layout |
| `composition(A, B)` | Apply B first, then use its scalar result as A's input |
| `coalesce(B)` | Simplify B while preserving its scalar mapping |
| `B.get_hier_coord(p)` | Recover a natural coordinate from an offset when the inverse helper's preconditions hold |
| A kernel using the layout | Assign accesses to threads; this is where hardware behavior must be measured |

For the later attention kernel, layouts describe how tiles and thread fragments address data. This entry establishes those maps with printed tables before adding tiling, cooperation, or a performance claim.

## Sources

[3] https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html

[4] https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-best-practices-guide/index.html

[5] https://github.com/NVIDIA/cutlass/blob/e94e888df3551224738bfa505787b515eae8352f/include/cute/layout.hpp

[6] https://github.com/NVIDIA/cutlass/blob/e94e888df3551224738bfa505787b515eae8352f/media/docs/cpp/cute/01_layout.md

[7] https://github.com/NVIDIA/cutlass/blob/e94e888df3551224738bfa505787b515eae8352f/media/docs/cpp/cute/02_layout_algebra.md
