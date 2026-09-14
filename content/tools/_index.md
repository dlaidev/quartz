---
title: "Interactive Tools"
description: "Browser examples for thread indexing and online softmax, with links to the underlying equations."
---

These browser examples illustrate indexing and arithmetic. They do not execute GPU kernels or predict device performance.

## CUDA thread and block indexing

The [CUDA visualizer](cuda-visualizer.md) shows how a one-dimensional launch maps threads to array elements. Change the array length and block size. Inspect the inactive tail of the final block and the thread groups within each block.

Read [CUDA threads and blocks](../2025-05-17-cuda-basics.md) for the indexing equations and the distinction between active lanes and hardware occupancy.

## Online softmax

<a href="/static/tensor-kernels/#online-softmax-demo" data-router-ignore>Open the online-softmax example</a> and advance through the score blocks. The display tracks the running maximum, exponential sum, and weighted sum. Rescaling the old sums preserves their meaning when a larger score arrives.

The [tensor-equations article](../2026-09-13-from-tensor-equations-to-fast-kernels.md) derives the update and includes a CPU verification script.

## Memory-layout examples

[CuTe layouts](../2025-05-10-cute-basics.md) explains shape-and-stride maps. [Triton linear layouts](../2025-06-22-linear-layouts.md) explains binary basis maps between execution coordinates and tensor coordinates. These articles contain worked examples; they describe different layout representations.

Use a profiler and the generated kernel when you need measured memory traffic, bank conflicts, or occupancy.
