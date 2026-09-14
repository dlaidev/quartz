---
title: "CUDA Thread/Block Visualizer"
date: 2025-05-17
tags:
  - CUDA
  - GPU-Programming
  - Interactive-Tools
  - Thread-Patterns
  - Visualization
draft: false
---

This browser tool draws a one-dimensional, one-element-per-thread mapping. It groups useful elements into blocks and warps. It does not run CUDA or measure a GPU.

The current implementation has known arithmetic and display limitations. In particular, **Warp Utilization** ignores block boundaries, and the drawing omits fully masked tail warps. The reference calculations below remain valid even when an interface label differs.

<div id="cuda-visualization"></div>

<script src="https://unpkg.com/react@18/umd/react.development.js"></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
<script src="/tools/cuda-visualizer.js"></script>

## Controls and symbols

| Control or symbol      | Current behavior                                   |
| ---------------------- | -------------------------------------------------- |
| Grid Size              | Useful element count N, from 1 to 512; default 128 |
| Block Size             | Threads per block B, from 1 to 256; default 32     |
| Block box              | One block; width reflects its useful element count |
| Warp row               | 32 lane dots; rows can wrap with available width   |
| Green dot              | An element assigned useful work under `i < N`      |
| Gray dot               | A displayed lane without useful work               |
| Block or warp hover    | Tooltip with its useful thread count               |
| Configuration Analysis | Toggle the statistics and heuristic labels         |

The **Grid Size** name differs from CUDA terminology. CUDA's grid dimensions count blocks.[3] In this tool, the number of blocks is `G = ceil(N/B)`. The interface does not allow N=0, multidimensional launches, grid-stride loops, or explicit grid dimensions independent of N.

Gray dots combine two cases: lanes that have no thread because B is not a multiple of 32, and launched threads whose index exceeds the data length. The final block still launches B threads.[3] Fully masked warps in that block are not drawn.

## What the analysis fields calculate

Let `L = N - (G-1)*B`, the number of useful elements in the final block.

| Field                 | Formula or interpretation                               |
| --------------------- | ------------------------------------------------------- |
| Blocks                | `ceil(N/B)`                                             |
| Block Efficiency      | `100*N/(G*B)`; useful fraction of launched threads      |
| Last Block Size       | L useful threads; actual block size remains B           |
| Last Block Efficiency | `100*L/B`                                               |
| Warps per Block       | `ceil(B/32)`                                            |
| Total Warps           | `G*ceil(B/32)`; includes warps missing from the drawing |
| Last Warp Size        | Useful lanes in the last warp containing useful work    |
| Warp Utilization      | `100*N/(32*ceil(N/32))`; ignores the chosen B           |

A block-aware useful-lane ratio for this mapping is:

$$
U = \frac{N}{32\,\lceil N/B\rceil\,\lceil B/32\rceil}.
$$

Neither U nor **Block Efficiency** is a measured execution efficiency. Occupancy instead concerns resident warps relative to an SM's supported maximum.[4]

## Worked configurations

The values below were checked with the [CPU reference program](/code/cuda-basics/checks.py). Every configuration fits within the current controls.

|   N |   B | Blocks | Launched warps | Useful thread fraction | Block-aware useful lane fraction |
| --: | --: | -----: | -------------: | ---------------------: | -------------------------------: |
| 256 |  32 |      8 |              8 |                 100.0% |                           100.0% |
| 100 |   7 |     15 |             15 |                  95.2% |                            20.8% |
| 129 |  32 |      5 |              5 |                  80.6% |                            80.6% |
| 129 | 128 |      2 |              8 |                  50.4% |                            50.4% |
| 512 | 256 |      2 |             16 |                 100.0% |                           100.0% |

For N=100, B=7, count fifteen partial warps. Each block occupies its own warp; CUDA does not merge threads from separate blocks.[3] The interface reports 78.1% **Warp Utilization**, while the block-aware ratio is 20.8%.

For N=129, B=128, count four full warp rows in the first block and one partial row in the second. The drawing shows five warps, but the launch has eight. Three additional warps in the second block fail the data bounds check for every lane. The interface reports 80.6% **Warp Utilization** instead of the block-aware 50.4%.

For N=512, compare B=32 and B=256. Both give sixteen warps with full coverage, but the number of independent blocks changes. The tool cannot decide which grouping is faster. See [[2025-05-17-cuda-visualizer|the exercise walkthrough]] for the derivation.

## Mapping to a kernel launch

For the N=129, B=128 exercise:

```cpp
__global__ void saxpy(size_t n, float a, const float *x, float *y)
{
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

// x and y must already be allocated and initialized for device access.
size_t n = 129;                         // "Grid Size" slider
unsigned blockSize = 128;               // "Block Size" slider
unsigned numBlocks = n / blockSize + (n % blockSize != 0);
saxpy<<<numBlocks, blockSize>>>(n, 2.0f, x, y);
```

The launch creates two blocks of 128 threads. The guard protects data accesses in the final block. A complete host program must check CUDA API and launch errors and wait for kernel completion before reading results.[3][4] This fragment documents the mapping; it was not compiled or run on a GPU here.

For an arbitrary large N, validate grid limits before narrowing the block count to an unsigned launch dimension. A grid-stride loop can process several elements per thread; its work distribution is outside this tool's model.

## Limits of the performance labels

The current **Memory Coalescing** label checks only `B % 32`. Coalescing requires the addresses used by the kernel's memory instructions, including element size and alignment.[4] Those inputs do not exist in the tool.

**Occupancy Impact** uses a block-size threshold. It has no GPU properties, register count, or shared-memory allocation. Its “Good” and “Moderate” classifications therefore have no calculated residency basis.

**Thread Divergence Risk** uses the tail size. It has no kernel control flow. A partial data tail can mask lanes, but branches within fully populated blocks can also diverge.[3] The label cannot predict a profiler's divergence counters.

The built-in recommendations for memory-bound and compute-bound block sizes are heuristics without measurements. Use [[2025-05-17-cuda-basics|CUDA threads and blocks]] for the distinctions between coverage, coalescing, bank conflicts, and occupancy.

## Verification and loading

The [component source](/tools/cuda-visualizer.js) supplies the behavior documented here. Its [Node harness](/code/cuda-basics/visualizer-checks.mjs) executes the component and checks its element tree, slider bounds, dot counts, and known discrepancies:

```sh
node content/code/cuda-basics/visualizer-checks.mjs
```

It ends with `Shipped visualizer behavior checks passed (known defects reproduced)`. That result verifies the documented implementation behavior rather than correcting its formulas. The companion Python checks supply the independent reference arithmetic.

The widget loads React and ReactDOM from an external CDN and mounts only on `DOMContentLoaded`. Client-side navigation may leave the container blank; reload the page directly in that case. A CDN or script-loading failure also prevents mounting. The tables and kernel mapping are available without JavaScript. Browser layout and GPU performance are outside the local element-tree test.

## Sources

[3] https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html

[4] https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-best-practices-guide/index.html
