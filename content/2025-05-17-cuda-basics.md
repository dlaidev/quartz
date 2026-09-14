---
title: "CUDA Threads and Blocks"
date: 2025-05-17
description: "Map CUDA threads to data, count partial warps, and distinguish launch coverage from occupancy."
tags: ["CUDA", "GPU Programming", "Parallel Computing", "Performance Optimization", "AI Inference"]
categories: ["CUDA", "GPU Optimization"]
showToc: false
TocOpen: true
---

A CUDA launch specifies a grid of blocks and a fixed shape for every block. Those dimensions determine thread IDs. The kernel maps thread IDs to data. This separation matters when counting tails, reasoning about memory addresses, and choosing a block size.

The [[tools/cuda-visualizer|thread/block visualizer]] draws a one-dimensional mapping. Its documentation lists several incorrect performance labels in the current interface. Use it to inspect grouping and compare the arithmetic below, rather than to predict execution time.

## Threads, warps, blocks, and grids

A thread executes one instance of a kernel. Threads in a block can exchange data through shared memory and synchronize with block barriers. Blocks form the grid passed to the launch.[3]

```text
Grid: G blocks
  Block 0: B threads, grouped into ceil(B/32) warps
  Block 1: B threads, grouped into ceil(B/32) warps
  ...
  Block G-1: B threads, grouped into ceil(B/32) warps
```

A warp contains 32 lanes. The hardware forms warps from consecutive linear thread IDs within a block; a warp never combines threads from different blocks.[3] If a block has seven threads, it occupies one partial warp. Fifteen such blocks contain fifteen warps, even though their combined thread count would fit into fewer full warps.

Warp instructions execute for their active lanes under CUDA's SIMT model. Threads can take different paths. Independent Thread Scheduling on Volta and later GPUs means code must use the required synchronization primitives rather than assume implicit lockstep for communication.[3] "Thirty-two threads simultaneously" is therefore an insufficient description of the execution model.

A block resides on one streaming multiprocessor (SM). An SM may hold several blocks if resources permit. Ordinary blocks must be independently schedulable, and block IDs do not specify an SM or an execution order.[3] Optional cluster and cooperative-launch features have additional rules; the examples here use ordinary launches.

## Map a one-dimensional launch

For `add<<<G,B>>>(...)`, the kernel computes:

```cpp
size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
```

`threadIdx.x` ranges from zero through `B-1`. `blockIdx.x` ranges from zero through `G-1`. `blockDim.x` is B; `gridDim.x` is G. The grid dimension counts blocks, rather than useful data elements.[3]

For a single element per thread and positive N:

$$
G = \lceil N/B \rceil,\qquad T = GB.
$$

Every block has B threads, including the final block. The last block's useful thread count is `N - (G-1)*B`. Extra threads must avoid out-of-bounds data access.

This kernel retains the original mapping record:

```cpp
struct ThreadInfo {
    size_t index;
    unsigned block;
    unsigned thread;
    float value;
};

__global__ void add(size_t n, const float *x, float *y, ThreadInfo *info)
{
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] += x[i];
        info[i] = {i, blockIdx.x, threadIdx.x, y[i]};
    }
}
```

The host supplies initialized, device-accessible arrays `x`, `y`, and `info`, each with N elements. For these small examples, the launch calculation is:

```cpp
unsigned b = 128;
size_t blocks = n / b + (n % b != 0);
if (n != 0)
    add<<<static_cast<unsigned>(blocks), b>>>(n, x, y, info);
```

The division form avoids overflow in `n + b - 1`. Before the cast, a general-purpose caller must check the device's grid limit. Skip the launch when N is zero. Check allocation and launch errors, synchronize before reading results on the host, and release allocations after use.[3][4]

The snippets explain the kernel interface; they are not a complete CUDA executable. Recording `ThreadInfo` adds memory traffic, so its timing would not describe a plain vector-add kernel. Unified Memory would also require a stated migration and timing policy.

## Two-dimensional blocks

Within a block, x varies fastest. The linear thread ID for a three-dimensional block is:[3]

```cpp
unsigned t = threadIdx.x
           + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
unsigned warp = t / 32;
unsigned lane = t % 32;
```

For `dim3 block(16,16)`, warp zero contains x=0..15 at y=0, followed by x=0..15 at y=1. For `dim3 block(32,8)`, warp zero covers x=0..31 at y=0. Both blocks have 256 threads and eight warps, but their mappings to matrix rows differ.

A row-major matrix kernel often maps x to the column:

```cpp
size_t col = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
size_t row = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
if (row < rows && col < cols)
    output[row * cols + col] = input[row * cols + col];
```

Ceiling-divide columns by `block.x` and rows by `block.y` to construct the grid. The row length and element size determine whether adjacent pieces of a warp's access touch adjacent memory sectors. A two-dimensional block shape alone cannot establish coalescing.

## Count useful threads and warp lanes separately

For these one-dimensional, single-pass kernels, define:

$$
W_b = \lceil B/32 \rceil,\quad W = G W_b,
\quad U_t = \frac{N}{GB},\quad U_l = \frac{N}{32W}.
$$

`Ut` is the fraction of launched threads assigned a valid element. `Ul` compares useful elements with all lane slots in the launch's warps. These are arithmetic coverage ratios, without any claim about elapsed cycles or SM residency.

|    N |   B | Blocks | Warps | Useful threads | Useful lane slots |
| ---: | --: | -----: | ----: | -------------: | ----------------: |
|  256 |  32 |      8 |     8 |         100.0% |            100.0% |
| 1024 | 256 |      4 |    32 |         100.0% |            100.0% |
|  100 |   8 |     13 |    13 |          96.2% |             24.0% |
|   97 |  32 |      4 |     4 |          75.8% |             75.8% |
|  100 |   7 |     15 |    15 |          95.2% |             20.8% |
|  129 | 128 |      2 |     8 |          50.4% |             50.4% |

The [CPU companion](/code/cuda-basics/checks.py) generated these values and verified coverage by enumerating each `(block,thread)` pair. Run:

```sh
uv run --no-project python content/code/cuda-basics/checks.py
```

### Full blocks: N=256, B=32

Eight blocks cover all elements. Each block contains one full warp. This establishes complete launch coverage. It does not establish high occupancy: the device may have more SMs than this grid has blocks, and one-warp blocks can encounter a per-SM block limit before a warp limit.

### Sub-warp blocks: N=100, B=8

The launch has thirteen blocks and 104 threads. Twelve blocks handle eight elements each; the final block handles four. Each block consumes one warp, so useful work covers only `100 / (13*32)` of the available lane slots. Unused lanes in a partial block warp cannot be filled by another block's threads.[3]

### A prime length: N=97, B=32

The first three blocks handle 96 elements. The final block launches 32 threads; only its lane zero passes the bounds check. Primality has no special hardware consequence here. The remainder determines the tail.

### A tail spanning several warps: N=129, B=128

Two blocks launch eight warps. In the second block, one lane in its first warp performs the addition; its other three warps have no valid elements. Those threads still execute enough of the kernel to evaluate the guard. A drawing that omits fully masked warps understates the launch footprint.

## Occupancy is a residency ratio

Occupancy is the ratio of resident warps on an SM to the maximum number of resident warps supported by that SM. Resident warps include warps waiting on dependencies. Occupancy is distinct from the fraction of lanes doing useful arithmetic.[4]

Limits include resident blocks, threads, warps, registers, and shared memory. The compiled kernel supplies register and static shared-memory requirements; the launch supplies dynamic shared memory. Allocation granularity and architecture-specific limits also matter.[3][4]

For a deliberately simplified resource model, assume an SM permits:

- 16 resident blocks;
- 2048 resident threads, or 64 warps;
- 65,536 32-bit registers;
- 65,536 bytes of shared memory.

Assume each 256-thread block uses eight warps, 32 registers per thread, and 16,384 bytes of shared memory. Ignore allocation rounding for this illustration. The block bounds are:

```text
block limit:          16
thread limit:         2048 / 256       = 8
warp limit:           64 / 8           = 8
register limit:       65536 / (32*256) = 8
shared-memory limit:  65536 / 16384    = 4
```

The minimum is four blocks. That permits 32 resident warps, or 50% occupancy in this model. The companion asserts these calculations. These invented resource limits explain the formula; they are not specifications or measurements of a named GPU.

For an actual device, query properties with `cudaGetDeviceProperties`. Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` with the compiled kernel, block size, and dynamic shared-memory byte count to estimate the residency bound.[3] The estimate assumes enough blocks are available. A short grid can leave SMs idle, and a profiler's achieved occupancy can be lower over the kernel's lifetime.

Higher occupancy does not necessarily improve performance. A kernel can already have enough active warps to hide latency, or an attempt to increase occupancy can cause register spilling and extra memory traffic.[4]

## Global-memory coalescing

For compute capability 6.0 and later, NVIDIA describes global-memory coalescing in terms of the 32-byte transactions needed to service a warp's addresses.[4] For one scalar float load per active lane and a suitably aligned base:

```text
lane:                 0    1    2    ...   31
contiguous element:   0    1    2    ...   31
stride-32 element:    0   32   64    ...  992
```

The contiguous case requests 128 bytes across four 32-byte sectors. Shifting the first float by one element spans five sectors. The stride-32 case touches 32 distinct sectors. These counts describe requested address coverage; cache hits and reuse affect traffic reaching device memory and elapsed time.[4]

A multiple-of-32 block size avoids structural partial warps. Coalescing still depends on the addresses used by each instruction. A warp-sized block can make scattered accesses, while a partial warp can access adjacent words efficiently within the sectors it touches.

## Shared-memory banks

Shared-memory bank conflicts are a different problem from global-memory coalescing. Under the standard 32-bank mapping for 32-bit words, bank selection is `word_index % 32`. Different words in the same bank can serialize a warp's request; reads of the same word can be broadcast.[4]

For one 32-bit shared-memory access per lane:

- `shared[lane]` addresses each bank once.
- `shared[2*lane]` addresses sixteen banks twice, giving two-way conflicts.
- `shared[32*lane]` addresses different words in one bank, giving a 32-way conflict.

A guarded vector add that uses only global memory cannot acquire a shared-memory bank conflict merely because N has a tail. Wider types and vector instructions require analysis of the actual accesses.

## Select and verify a launch

Start with a legal block shape that matches data indexing and cooperation. A multiple of the warp size is a useful candidate; power-of-two sizes are not a general requirement. Query per-block and per-dimension limits rather than assume every kernel can launch 1024 threads.[3][4]

Test several candidates with the same problem and input data. Check every output before timing. Record compiler resource usage and keep setup costs separate from kernel timing. A launch with fewer blocks can reduce available device-wide parallelism even when every block is full.

For a grid-stride loop, each thread visits `i`, then `i + blockDim.x*gridDim.x`, and so on. That allows the grid size to be chosen independently of the number of elements. It changes the single-pass coverage model used in the table and visualizer.

The local checks validate indexing, tails, sector counts, and the illustrative occupancy calculation. GPU compilation, execution, and timing remain separate tests; this article reports none of those measurements.

## Sources

[3] https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html

[4] https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-best-practices-guide/index.html
