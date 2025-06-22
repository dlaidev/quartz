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

# CUDA Thread/Block Visualizer

This interactive tool helps you understand how different thread and block configurations affect GPU execution patterns.

<div id="cuda-visualizer-root"></div>

<script src="https://unpkg.com/react@18/umd/react.development.js"></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>

<script>
  document.addEventListener('DOMContentLoaded', () => {
    const { useState } = React;

    // CUDA Visualization Component
    const CUDAVisualization = () => {
      const [gridSize, setGridSize] = useState(128);
      const [blockSize, setBlockSize] = useState(32);
      const [showDetails, setShowDetails] = useState(false);
      
      const numBlocks = Math.ceil(gridSize / blockSize);
      const lastBlockSize = gridSize % blockSize || blockSize;
      const warpSize = 32;
      const warpsPerBlock = Math.ceil(blockSize / warpSize);
      const lastBlockWarps = Math.ceil(lastBlockSize / warpSize);
      const lastWarpSize = lastBlockSize % warpSize || (lastBlockSize < warpSize ? lastBlockSize : warpSize);

      // Simple chevron icons (replacing lucide-react)
      const ChevronDown = ({ size = 20 }) => (
        React.createElement('svg', { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" },
          React.createElement('polyline', { points: "6,9 12,15 18,9" })
        )
      );
      
      const ChevronUp = ({ size = 20 }) => (
        React.createElement('svg', { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" },
          React.createElement('polyline', { points: "18,15 12,9 6,15" })
        )
      );

      const renderBlocks = () => {
        const blocks = [];
        for (let b = 0; b < numBlocks; b++) {
          const isLastBlock = b === numBlocks - 1;
          const currentBlockSize = isLastBlock ? lastBlockSize : blockSize;
          const currentWarps = isLastBlock ? lastBlockWarps : warpsPerBlock;
          
          const warps = [];
          for (let w = 0; w < currentWarps; w++) {
            const isLastWarpOfBlock = w === currentWarps - 1;
            const currentWarpSize = isLastBlock && isLastWarpOfBlock 
              ? lastWarpSize 
              : Math.min(warpSize, currentBlockSize - w * warpSize);

            const threads = [];
            for (let t = 0; t < warpSize; t++) {
              const isActiveThread = t < currentWarpSize;
              threads.push(React.createElement('div', { 
                key: t, 
                style: {
                  width: '8px', height: '8px', margin: '1px', borderRadius: '50%',
                  backgroundColor: isActiveThread ? '#22c55e' : '#64748b',
                  opacity: isActiveThread ? 1 : 0.3
                }
              }));
            }
            warps.push(React.createElement('div', { key: w, style: { display: 'flex', flexWrap: 'wrap', padding: '2px', margin: '2px', backgroundColor: '#0f172a', borderRadius: '3px' }, title: `Warp ${b}-${w}: ${currentWarpSize}/${warpSize} threads` }, threads));
          }
          
          blocks.push(
            React.createElement('div', { key: b, style: { display: 'flex', flexDirection: 'column', border: '2px solid #334155', borderRadius: '4px', margin: '4px', backgroundColor: '#1e293b', width: `${Math.max(100, (currentBlockSize / blockSize) * 200)}px`, overflow: 'hidden' }, title: `Block ${b}: ${currentBlockSize} threads` }, [
              React.createElement('div', { key: 'header', style: { backgroundColor: '#334155', color: 'white', padding: '2px 6px', textAlign: 'center', fontSize: '12px' } }, `Block ${b}`),
              React.createElement('div', { key: 'warps', style: { padding: '2px' } }, warps)
            ])
          );
        }
        return blocks;
      };

      return React.createElement('div', { style: { padding: '16px', backgroundColor: '#f3f4f6', borderRadius: '8px', fontFamily: 'sans-serif' } }, [
        React.createElement('h2', { key: 'title', style: { fontSize: '20px', fontWeight: 'bold', marginBottom: '16px' } }, 'Interactive CUDA Thread/Block Visualizer'),
        React.createElement('div', { key: 'controls', style: { marginBottom: '16px' } }, [
          React.createElement('div', { key: 'grid-slider', style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' } }, [
            React.createElement('label', { style: { fontWeight: '500' } }, `Grid Size: ${gridSize} threads`),
            React.createElement('input', { type: 'range', min: '1', max: '1024', value: gridSize, onChange: (e) => setGridSize(parseInt(e.target.value)), style: { width: '200px' } })
          ]),
          React.createElement('div', { key: 'block-slider', style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' } }, [
            React.createElement('label', { style: { fontWeight: '500' } }, `Block Size: ${blockSize} threads`),
            React.createElement('input', { type: 'range', min: '1', max: '512', value: blockSize, onChange: (e) => setBlockSize(parseInt(e.target.value)), style: { width: '200px' } })
          ])
        ]),
        React.createElement('div', { key: 'viz', style: { marginBottom: '16px', backgroundColor: 'white', padding: '16px', borderRadius: '4px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', minHeight: '100px' } }, 
          React.createElement('div', { style: { display: 'flex', flexWrap: 'wrap' } }, renderBlocks())
        ),
        React.createElement('div', { key: 'analysis', style: { backgroundColor: 'white', padding: '16px', borderRadius: '4px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', marginBottom: '16px' } }, [
          React.createElement('div', { key: 'header', onClick: () => setShowDetails(!showDetails), style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' } }, [
            React.createElement('h3', { style: { fontWeight: 'bold' } }, 'Configuration Analysis'),
            showDetails ? React.createElement(ChevronUp) : React.createElement(ChevronDown)
          ]),
          showDetails && React.createElement('div', { key: 'details', style: { marginTop: '12px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '14px' } }, [
            React.createElement('span', null, 'Total Blocks:'), React.createElement('span', null, numBlocks),
            React.createElement('span', null, 'Threads in Last Block:'), React.createElement('span', null, `${lastBlockSize} / ${blockSize}`),
            React.createElement('span', null, 'Total Warps:'), React.createElement('span', null, numBlocks * warpsPerBlock),
            React.createElement('span', null, 'Warps in Last Block:'), React.createElement('span', null, lastBlockWarps),
          ])
        ]),
      ]);
    };

    const container = document.getElementById('cuda-visualizer-root');
    if (container) {
      ReactDOM.render(React.createElement(CUDAVisualization), container);
    }
  });
</script>

## How to Use

1. **Grid Size**: Total number of threads in your kernel (adjust with top slider)
2. **Block Size**: Number of threads per block (adjust with bottom slider)
3. **Green dots**: Active threads executing work
4. **Gray dots**: Inactive threads (padding within warps)
5. **Click "Configuration Analysis"** to see detailed performance metrics

## Key Insights

**Block size should be a multiple of 32** ([[Warp Size]]) for optimal [[Memory Coalescing]]
- Warps execute 32 threads simultaneously
- Non-aligned block sizes waste GPU resources
- Memory transactions are most efficient with aligned access

**Larger blocks** reduce scheduling overhead but may limit [[GPU Occupancy]]
- More threads per block = fewer blocks
- Better for compute-intensive kernels with thread cooperation
- May hit resource limits (registers, shared memory)

**Power-of-2 sizes** often work best for [[Memory Access Patterns]]
- Align with cache line boundaries
- Simplify address calculations
- Reduce memory bank conflicts

## Experiment with These Configurations

Try these scenarios to understand the trade-offs:

**Perfect Configuration**: Grid=256, Block=32
- 8 blocks × 32 threads = 8 warps total
- 100% efficiency, perfect alignment

**Poor Configuration**: Grid=100, Block=7  
- 15 blocks with irregular warp usage
- Significant thread waste in each warp

**Large Block**: Grid=1024, Block=256
- 4 blocks × 256 threads = 32 warps total  
- Good for thread cooperation, may limit occupancy

## Related Concepts

- [[CUDA Architecture Deep Dive]]
- [[GPU Memory Hierarchy]] 
- [[Warp Execution Patterns]]
- [[SM Resource Management]]
