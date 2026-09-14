---
title: "Triton Linear Layouts"
date: 2025-06-22
description: "Triton's GF(2) layout maps, with tested register-to-tensor coordinates, XOR composition, and a CPU coalescing model."
tags:
  - Triton
  - Linear-Layouts
  - GPU-Programming
  - Memory-Management
  - Performance-Optimization
  - AI
  - Compilers
  - Backend
  - Parallel-Computing
  - AI-Inference
draft: false
---

A tensor tile has logical coordinates. A GPU distributes its elements across registers and threads. Triton's linear layouts describe the map from those hardware locations to tensor coordinates. Their arithmetic is linear over **GF(2)**: basis contributions combine with bitwise XOR.[1]

This differs from a conventional shape-and-stride map. Separating the two maps helps explain tensor ownership, layout conversion, and memory coalescing without confusing a register location with a byte address.

The source references use Triton **v3.3.1**, commit `d654e0f2d91f07496454e0fcbec2a9b97df37d47`. The CPU companion implements the mathematics independently; it does not call Triton's C++ implementation or measure GPU performance.

## Shape and stride describe a different map

A basic CuTe layout maps a coordinate to an index using its shape and stride. For a row-major tensor with shape `(4, 32)` and stride `(32, 1)`, the natural-coordinate map is:[4]

$$
f(r,c) = 32r + c.
$$

For four-byte elements, an address calculation then gives `base + 4*f(r,c)`. CuTe also supports hierarchical coordinates and layout algebra; the formula above describes this particular flat layout, rather than all of CuTe.[4]

Triton's `LinearLayout` typically maps `(register, lane, warp, block)` to `(row, column)`. Shared-memory layouts can instead map `(offset, block)` to tensor coordinates. The hardware-to-logical direction permits multiple locations to hold the same element while keeping the map a function.[1]

The two maps answer separate questions:

```text
(register, lane, warp, block)
          | tensor ownership layout
          v
      (row, column)
          | storage strides and element size
          v
       byte address
```

The coordinate-to-address calculation can use ordinary integer arithmetic. The ownership layout uses XOR on coordinate bits. In selected power-of-two cases, ordinary addition and XOR happen to agree because the contributing bits do not overlap. That coincidence does not make the two algebras interchangeable.

## What is linear over GF(2)?

GF(2) contains the bits zero and one. Addition is XOR; multiplication is AND. Treat each input coordinate as a vector of bits. Each input bit has an output basis vector. To evaluate the layout, XOR the basis vectors for the input bits that are set.[1]

For input bits $x_j$ and basis columns $B_j$:

$$
L(x) = \bigoplus_j x_j B_j.
$$

The coefficient $x_j$ selects either the zero vector or $B_j$. Each component of an output basis vector encodes bits of one tensor coordinate. Thus `(1, 4)` contributes row bit zero and column bit two. It does not mean an integer stride of four after an integer displacement of one.

Linearity requires:

$$
L(x \oplus y) = L(x) \oplus L(y), \qquad L(0)=0.
$$

A fixed nonzero translation requires a separate offset or an affine extension. Integer addition with carries generally violates this XOR rule.

For example, the map `S(row, col) = (row, col XOR row)` is linear over GF(2). Applying it twice recovers the input, since XORing the same row twice cancels it. Triton's header uses a related swizzle to introduce basis evaluation.[1]

## A register tile, bit by bit

Construct a `(4, 32)` tile using two warps of 32 lanes, with two elements per lane. This is a mathematical register-distribution example. We do not claim that Triton selects this layout for every operation or target.

The input coordinates have these meanings:

| Input      | Range   | Meaning                                        |
| ---------- | ------- | ---------------------------------------------- |
| `register` | `0..1`  | Element slot within one thread's tile fragment |
| `lane`     | `0..31` | Thread within a warp                           |
| `warp`     | `0..1`  | Warp within this tile's thread block           |
| `block`    | `0`     | Singleton block dimension in this example      |

`register` is an abstract element-slot index, not a physical register number assigned by the backend. A nontrivial `block` dimension can express distribution across CTAs, including cluster layouts. It is not automatically the global `program_id` grid coordinate. This example handles one tile; a kernel's global tile offset belongs to its indexing logic.[1]

Choose these bases, in low-bit-first order within each input dimension:

| Input bit set  | Output `(row, column)` |
| -------------- | ---------------------- |
| `register = 1` | `(0, 16)`              |
| `lane = 1`     | `(0, 1)`               |
| `lane = 2`     | `(0, 2)`               |
| `lane = 4`     | `(0, 4)`               |
| `lane = 8`     | `(0, 8)`               |
| `lane = 16`    | `(1, 0)`               |
| `warp = 1`     | `(2, 0)`               |

The singleton `block` dimension has no input bits and contributes nothing. The same map has the explicit formula:

```python
row = (lane >> 4) ^ (warp << 1)
col = (lane & 15) ^ (register << 4)
```

These particular contributions occupy disjoint bits, so addition would give the same result. A later swizzle will introduce overlapping contributions.

For `(register, lane, warp, block) = (1, 19, 1, 0)`, lane 19 has bits 16, 2, and 1 set. Select the corresponding bases:

```text
(0,16) XOR (1,0) XOR (0,2) XOR (0,1) XOR (2,0) = (3,19)
```

Selected locations:

| Register | Lane | Warp | Tensor coordinate |
| -------- | ---- | ---- | ----------------- |
| 0        | 0    | 0    | `(0, 0)`          |
| 0        | 15   | 0    | `(0, 15)`         |
| 0        | 16   | 0    | `(1, 0)`          |
| 1        | 0    | 0    | `(0, 16)`         |
| 1        | 19   | 1    | `(3, 19)`         |

The companion enumerates every hardware tuple and compares the output set with the independent Cartesian product `range(4) × range(32)`. It verifies both coverage and unique ownership. A count alone would not detect a map that covers the wrong coordinates.

## Inversion and replication

This example is bijective. Its inverse can recover each input field directly:

```python
register = col >> 4
lane = (col & 15) | ((row & 1) << 4)
warp = row >> 1
block = 0
```

The companion also constructs an inverse lookup table by enumeration and checks every round trip. Triton's implementation represents layouts as bit matrices and uses algebraic inversion machinery; exhaustive lookup is suitable only for this small teaching example.[2]

A general linear layout need not be injective. A zero basis means that toggling that input bit leaves the tensor coordinate unchanged, so two hardware locations own the same element. Dependent nonzero bases can also produce duplicates. In matrix terms, a nontrivial kernel prevents a unique inverse. A surjective layout still covers every output coordinate, but choosing one preimage requires a convention.[1]

Triton's `invertAndCompose` supports this case. If `C = A.invertAndCompose(B)`, the documented relationship is `B(C(x)) = A(x)`. When `B` is invertible, this becomes `C = B^-1 ∘ A`. Otherwise the implementation chooses a compatible preimage under its documented requirements; it does not recover a unique original hardware location.[1][2]

## Composition and layout conversion

Let $R$ map register locations to tensor coordinates, and let $S$ map shared-memory offsets to the same tensor coordinates. The store-address conversion is:

$$
C = S^{-1} \circ R.
$$

It maps a register location to the shared-memory offset that holds the same logical element. In Triton's API this is `R.invertAndCompose(S)`. Dimension names and sizes must be compatible, and the outer layout must satisfy the documented coverage requirements.[1]

For ordinary composition, `A.compose(B)` means apply `A` first and `B` second: $B \circ A$. This order is easy to reverse accidentally.[1]

Apply the XOR swizzle `S(row, col) = (row, col XOR row)` to our register layout. Composition can be computed by applying `S` to every output basis of `R`. The column-only bases stay unchanged. The two row-producing bases change:

```text
lane = 16: (1,0) -> (1,1)
warp = 1:  (2,0) -> (2,2)
```

The composed map sends `(1,19,1,0)` to `(3,16)`, because `19 XOR 3 = 16`. Integer addition would give a different answer. The tests compare basis composition with direct function composition for every hardware tuple.

A conversion map identifies where each value must come from. Lowering still has to implement the transfer. A change confined to register slots within a thread may need only local rearrangement. A change of lane ownership may require shuffles or shared memory. Transfers between warps need mechanisms with the appropriate scope. Algebraic invertibility alone does not establish that the conversion is cheap or requires no synchronization.

## Coalescing needs addresses for one instruction

Coalescing depends on the addresses requested by participating lanes. Looking only at a tensor's logical shape or at one lane's sequence of elements is insufficient.

Hold `register=0` and `warp=0` fixed in the example. Suppose every lane issues one four-byte load, every lane is active, and the base address is aligned to 32 bytes. With row-major storage, lanes 0–15 access row 0, columns 0–15. Lanes 16–31 access row 1, columns 0–15.

The byte ranges are `0..63` and `128..191`. They touch four aligned 32-byte sectors. The range across the whole warp has a gap, but the individual sectors contain no gaps in the requested four-byte elements.

With column-major storage of the same `(4,32)` tensor, use `base + 4*(row + 4*col)`. The identical ownership layout now touches eight 32-byte sectors. Moving the row-major base forward by four bytes increases its touched-sector count to six. The companion computes:

| Byte base | Storage formula in elements | Touched 32-byte sectors |
| --------- | --------------------------- | ----------------------- |
| 0         | `32*row + col`              | 4                       |
| 0         | `row + 4*col`               | 8                       |
| 4         | `32*row + col`              | 6                       |
| 4         | `row + 4*col`               | 8                       |

This is an address-set model with an explicitly chosen sector size. It is not a measured count of DRAM transactions, cache misses, or generated load instructions. Vectorization, masks, caching, instruction grouping, and architecture can change those outcomes. A shared-memory bank analysis requires bank mapping and access-width rules instead of this global-memory sector model.

## Run and check the model

Download [linear_layout.py](code/triton-analysis/linear_layout.py), [dataflow.py](code/triton-analysis/dataflow.py), and [test_analysis.py](code/triton-analysis/test_analysis.py) into one directory. They use only the Python 3 standard library. From the repository root:

```bash
uv run --no-project python -B content/code/triton-analysis/linear_layout.py
uv run --no-project python -B -m unittest discover -s content/code/triton-analysis -v
```

Output from the executed layout program:

```text
locations=128 unique_coordinates=128
L(1, 19, 1, 0)=(3, 19)
inverse(3, 19)=(1, 19, 1, 0)
swizzled(1, 19, 1, 0)=(3, 16)
base=0 row-major: touched_32B_sectors=4
base=0 column-major: touched_32B_sectors=8
base=4 row-major: touched_32B_sectors=6
base=4 column-major: touched_32B_sectors=8
```

All 11 tests in the combined dataflow/layout suite passed. Layout tests check coverage against the Cartesian product, the explicit coordinate formula, inverse round trips, XOR linearity for every pair of inputs, composition, sector counts, and invalid input rejection. Triton's own `LinearLayoutTest.cpp` provides separate upstream examples and tests of the actual C++ API.[3]

## Limits of the representation

A linear layout has power-of-two bit domains. Irregular problem extents need surrounding tiling and validity masks; a layout map alone does not establish which elements of a boundary tile are valid. Arbitrary gathers and data-dependent permutations generally cannot be represented by a fixed GF(2) matrix.[1]

The map also leaves several decisions outside its scope:

- A storage base pointer, element type, and strides are needed to form byte addresses.
- Replication affects inverse selection and can require redundant-store suppression.
- Matrix instructions constrain legal operand distributions. A mathematically valid layout is not necessarily accepted by a particular instruction.
- Physical register allocation, occupancy, and instruction scheduling occur beyond this coordinate model.

Use the bases to establish ownership and conversion. Use the emitted instructions and a stated memory model to analyze accesses. Measure a kernel before making a performance claim.

## Sources

[1] https://raw.githubusercontent.com/triton-lang/triton/d654e0f2d91f07496454e0fcbec2a9b97df37d47/include/triton/Tools/LinearLayout.h — include/triton/Tools/LinearLayout.h

[2] https://raw.githubusercontent.com/triton-lang/triton/d654e0f2d91f07496454e0fcbec2a9b97df37d47/lib/Tools/LinearLayout.cpp — lib/Tools/LinearLayout.cpp

[3] https://raw.githubusercontent.com/triton-lang/triton/d654e0f2d91f07496454e0fcbec2a9b97df37d47/unittest/Tools/LinearLayoutTest.cpp — unittest/Tools/LinearLayoutTest.cpp

[4] https://docs.nvidia.com/cutlass/media/docs/cpp/cute/01_layout.html — CuTe layout tutorial
