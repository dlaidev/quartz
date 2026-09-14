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

The visualizer groups a one-dimensional problem into blocks and warps. It runs JavaScript in the browser and does not execute a CUDA kernel. Use it to inspect integer indexing and tail coverage.

The current interface also displays heuristic performance labels and an incorrect aggregate warp-utilization formula. The exercises below state the correct counts and identify where the display differs. Full control definitions are on the [[tools/cuda-visualizer|tool reference page]].

<div id="cuda-visualization"></div>

<script src="https://unpkg.com/react@18/umd/react.development.js"></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
<script src="/tools/cuda-visualizer.js"></script>

## Read the drawing

The top slider is labeled **Grid Size**, but it sets N, the number of useful elements. Its range is 1–512. CUDA's `gridDim.x` counts blocks; the corresponding value is `ceil(N/B)`, where B is the lower slider's block size.[3] B ranges from 1–256. The default is N=128 and B=32.

Each block box contains rows of 32 dots. Green dots represent elements that pass a conceptual `i < N` guard. Gray dots can represent either unused lanes beyond the block's thread count or launched threads that fail the guard. Those cases differ on hardware. The drawing also omits fully masked warps at the end of the last block.

Hover over a block or warp to read its tooltip. Click **Configuration Analysis** to expand the arithmetic and labels. The counts describe a single-pass mapping:

```cpp
size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
if (i < n)
    output[i] = input[i];
```

All blocks in an ordinary launch have the same thread count.[3] A smaller last box depicts fewer useful elements; it does not mean CUDA launches a smaller last block.

## Exercise 1: full blocks

Set N=256 and B=32.

```text
blocks = ceil(256/32) = 8
warps per block = 1
total warps = 8
useful lane slots = 256/(8*32) = 100.0%
```

The drawing has eight full warps. The absence of a tail establishes complete coverage. It does not establish occupancy or memory throughput. Eight blocks may be too few to use every SM on a particular device.

## Exercise 2: separate block fill from warp fill

Set N=100 and B=7. Fourteen blocks handle seven elements each, and the fifteenth handles two.

```text
launched threads = 15*7 = 105
launched warps = 15*ceil(7/32) = 15
useful threads = 100/105 = 95.2%
useful lane slots = 100/(15*32) = 20.8%
```

The current **Warp Utilization** field says 78.1%. It computes `N / (32*ceil(N/32))` and ignores block boundaries. CUDA cannot combine partial warps from separate blocks.[3] The fifteen drawn warps give a better basis for this calculation than that percentage.

## Exercise 3: distinguish a tail lane from omitted warps

Set N=129 and B=32. Five blocks launch five warps. The final warp has one useful lane, so useful lane-slot coverage is 80.6%.

Keep N=129 and change B to 128. CUDA now launches two blocks of four warps each. The first block handles 128 elements. Only one lane in the second block performs useful work.

```text
launched warps = 2*4 = 8
useful lane slots = 129/(8*32) = 50.4%
```

The interface draws five warps because it omits the three fully masked warps in the second block. **Total Warps** correctly reports eight, while **Warp Utilization** still reports 80.6%. This discrepancy is a display limitation, rather than a CUDA optimization that removes those threads from the launch.

## Exercise 4: change grouping without changing coverage

Set N=512 and compare B=32 with B=256. Both launches have sixteen warps and full coverage. The first uses sixteen blocks; the second uses two. Threads in a block can share memory and use block barriers, so the grouping affects cooperation and the number of independently schedulable blocks.[3]

The tool has no device model or kernel resource data. It cannot determine which configuration runs faster. The old N=1024 and N=4096 exercises lie outside the slider range; their arithmetic belongs in [[2025-05-17-cuda-basics|the CUDA tutorial]].

## Interpret the performance labels cautiously

**Memory Coalescing** tests only whether B is divisible by 32. Actual coalescing depends on the addresses requested by active lanes in each memory instruction.[4]

**Occupancy Impact** classifies blocks of up to 128 threads as “Good” and larger available blocks as “Moderate.” Actual occupancy depends on the device, compiled register use, shared memory, and residency limits.[4]

**Thread Divergence Risk** uses the final block's useful-thread count. It does not inspect branches or distinguish branch instructions from predication. Threads may diverge inside a fully populated block; a tail alone cannot describe all control flow in a kernel.[3]

The built-in block-size tips are unmeasured heuristics. Neither dot colors nor percentage labels report elapsed time, bandwidth, cache behavior, or achieved occupancy.

## Reproduce the checks

From the repository root:

```sh
uv run --no-project python content/code/cuda-basics/checks.py
node content/code/cuda-basics/visualizer-checks.mjs
```

The Python program enumerates the launch mapping and checks the arithmetic independently. The Node program executes the shipped [visualizer component](/tools/cuda-visualizer.js) with a minimal React element-tree harness. It verifies slider ranges and dot counts, and reproduces the discrepancies above. It does not test browser rendering, CDN availability, or GPU behavior.

If the controls remain blank after navigating within the site, reload this page directly. The script mounts on `DOMContentLoaded`, which does not repeat for client-side navigation. The static exercises remain usable when the widget cannot mount.

## Sources

[3] https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html

[4] https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-best-practices-guide/index.html
