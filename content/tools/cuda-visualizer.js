document.addEventListener('DOMContentLoaded', () => {
  // Only run if React is available and we're on the CUDA visualizer page
  if (typeof React === 'undefined' || typeof ReactDOM === 'undefined') {
    console.log('React not loaded, skipping CUDA visualizer');
    return;
  }

  const container = document.getElementById('cuda-visualization');
  if (!container) {
    console.log('CUDA visualizer container not found');
    return;
  }

  const { useState } = React;

  // CUDA Visualization Component
  const CUDAVisualization = () => {
    const [gridSize, setGridSize] = useState(128);
    const [blockSize, setBlockSize] = useState(32);
    const [showDetails, setShowDetails] = useState(false);
    
    const numBlocks = Math.ceil(gridSize / blockSize);
    const blockEfficiency = (gridSize / (blockSize * numBlocks)) * 100;
    const lastBlockSize = gridSize % blockSize || blockSize;
    const lastBlockEfficiency = (lastBlockSize / blockSize) * 100;
    const warpSize = 32;
    const warpsPerBlock = Math.ceil(blockSize / warpSize);
    const totalWarps = numBlocks * warpsPerBlock;
    const lastBlockWarps = Math.ceil(lastBlockSize / warpSize);
    const lastWarpSize = lastBlockSize % warpSize || (lastBlockSize < warpSize ? lastBlockSize : warpSize);
    const lastWarpEfficiency = (lastWarpSize / warpSize) * 100;
    
    // Calculate warp utilization
    const fullWarps = Math.floor(gridSize / warpSize);
    const partialWarpThreads = gridSize % warpSize;
    const warpUtilization = partialWarpThreads === 0 
      ? 100 
      : ((fullWarps * warpSize + partialWarpThreads) / ((fullWarps + 1) * warpSize)) * 100;

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
          const isLastWarp = isLastBlock && w === currentWarps - 1 && lastWarpSize < warpSize;
          const currentWarpSize = isLastWarp ? lastWarpSize : Math.min(warpSize, currentBlockSize - w * warpSize);

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
          
          warps.push(
            React.createElement('div', { 
              key: w, 
              style: { 
                display: 'flex', flexWrap: 'wrap', padding: '2px', margin: '2px', 
                backgroundColor: '#0f172a', borderRadius: '3px', position: 'relative' 
              },
              title: `Warp ${b}-${w}: ${currentWarpSize}/${warpSize} threads`
            }, 
            React.createElement('div', { style: { display: 'flex', flexWrap: 'wrap' } }, threads))
          );
        }
        
        blocks.push(
          React.createElement('div', { 
            key: b, 
            style: { 
              display: 'flex', flexDirection: 'column', border: '2px solid #334155', 
              borderRadius: '4px', margin: '4px', backgroundColor: '#1e293b', 
              width: `${Math.max(100, (currentBlockSize / blockSize) * 200)}px`, overflow: 'hidden' 
            }, 
            title: `Block ${b}: ${currentBlockSize} threads`
          }, [
            React.createElement('div', { 
              key: 'header', 
              style: { backgroundColor: '#334155', padding: '2px 6px', textAlign: 'center', fontSize: '12px', color: 'white' } 
            }, `Block ${b}`),
            React.createElement('div', { key: 'warps' }, warps)
          ])
        );
      }
      
      return blocks;
    };

    return React.createElement('div', { 
      style: { padding: '16px', backgroundColor: '#f3f4f6', borderRadius: '8px', marginBottom: '20px', fontFamily: 'sans-serif' }
    }, [
      React.createElement('h2', { 
        key: 'title',
        style: { fontSize: '20px', fontWeight: 'bold', marginBottom: '16px' }
      }, 'Interactive CUDA Thread/Block Visualizer'),
      
      React.createElement('div', { key: 'controls', style: { marginBottom: '16px' } }, [
        React.createElement('div', { 
          key: 'grid-control',
          style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }
        }, [
          React.createElement('label', { 
            key: 'grid-label',
            style: { fontWeight: '500' }
          }, `Grid Size: ${gridSize} threads`),
          React.createElement('input', {
            key: 'grid-input',
            type: 'range',
            min: '1',
            max: '512',
            value: gridSize,
            onChange: (e) => setGridSize(parseInt(e.target.value)),
            style: { width: '192px' }
          })
        ]),
        
        React.createElement('div', { 
          key: 'block-control',
          style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' }
        }, [
          React.createElement('label', { 
            key: 'block-label',
            style: { fontWeight: '500' }
          }, `Block Size: ${blockSize} threads`),
          React.createElement('input', {
            key: 'block-input',
            type: 'range',
            min: '1',
            max: '256',
            value: blockSize,
            onChange: (e) => setBlockSize(parseInt(e.target.value)),
            style: { width: '192px' }
          })
        ])
      ]),
      
      React.createElement('div', { 
        key: 'visualization',
        style: { marginBottom: '16px', backgroundColor: 'white', padding: '16px', borderRadius: '4px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }
      }, React.createElement('div', { style: { display: 'flex', flexWrap: 'wrap' } }, renderBlocks())),
      
      React.createElement('div', { 
        key: 'analysis',
        style: { backgroundColor: 'white', padding: '16px', borderRadius: '4px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', marginBottom: '16px' }
      }, [
        React.createElement('div', { 
          key: 'analysis-header',
          style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' },
          onClick: () => setShowDetails(!showDetails)
        }, [
          React.createElement('h3', { key: 'analysis-title', style: { fontWeight: 'bold' } }, 'Configuration Analysis'),
          showDetails ? React.createElement(ChevronUp, { key: 'chevron', size: 20 }) : React.createElement(ChevronDown, { key: 'chevron', size: 20 })
        ]),
        
        showDetails && React.createElement('div', { 
          key: 'analysis-details',
          style: { marginTop: '12px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '14px' }
        }, [
          React.createElement('div', { 
            key: 'block-stats',
            style: { backgroundColor: '#f9fafb', padding: '8px', borderRadius: '4px' }
          }, [
            React.createElement('p', { key: 'blocks' }, React.createElement('strong', null, 'Blocks: '), numBlocks),
            React.createElement('p', { key: 'block-eff' }, React.createElement('strong', null, 'Block Efficiency: '), `${blockEfficiency.toFixed(1)}%`),
            React.createElement('p', { key: 'last-block-size' }, React.createElement('strong', null, 'Last Block Size: '), `${lastBlockSize} threads`),
            React.createElement('p', { key: 'last-block-eff' }, React.createElement('strong', null, 'Last Block Efficiency: '), `${lastBlockEfficiency.toFixed(1)}%`)
          ]),
          
          React.createElement('div', { 
            key: 'warp-stats',
            style: { backgroundColor: '#f9fafb', padding: '8px', borderRadius: '4px' }
          }, [
            React.createElement('p', { key: 'warps-per-block' }, React.createElement('strong', null, 'Warps per Block: '), warpsPerBlock),
            React.createElement('p', { key: 'total-warps' }, React.createElement('strong', null, 'Total Warps: '), totalWarps),
            React.createElement('p', { key: 'last-warp-size' }, React.createElement('strong', null, 'Last Warp Size: '), `${lastWarpSize} threads`),
            React.createElement('p', { key: 'warp-util' }, React.createElement('strong', null, 'Warp Utilization: '), `${warpUtilization.toFixed(1)}%`)
          ]),
          
          React.createElement('div', { 
            key: 'performance-analysis',
            style: { backgroundColor: '#f9fafb', padding: '8px', borderRadius: '4px', marginTop: '8px', gridColumn: 'span 2' }
          }, [
            React.createElement('p', { key: 'memory-coalescing' }, 
              React.createElement('strong', null, 'Memory Coalescing: '),
              blockSize % 32 === 0 ? 
                "Optimal (block size is a multiple of warp size)" : 
                "Sub-optimal (block size is not a multiple of warp size)"
            ),
            React.createElement('p', { key: 'occupancy' }, 
              React.createElement('strong', null, 'Occupancy Impact: '),
              blockSize <= 128 ? "Good (smaller blocks allow more concurrent blocks)" :
              blockSize <= 256 ? "Moderate (medium blocks balance resources)" :
              "Low (large blocks may limit concurrency)"
            ),
            React.createElement('p', { key: 'divergence' }, 
              React.createElement('strong', null, 'Thread Divergence Risk: '),
              gridSize % blockSize === 0 ? "Low (even distribution of threads)" :
              lastBlockSize < blockSize / 2 ? "High (last block significantly underutilized)" :
              "Moderate (some underutilization in last block)"
            )
          ])
        ])
      ]),
      
      React.createElement('div', { 
        key: 'tips',
        style: { backgroundColor: '#dbeafe', padding: '16px', borderRadius: '4px', fontSize: '14px' }
      }, [
        React.createElement('p', { key: 'tips-title', style: { fontWeight: 'bold' } }, 'Tips for Optimal Configuration:'),
        React.createElement('ul', { key: 'tips-list', style: { listStyleType: 'disc', paddingLeft: '20px', marginTop: '8px' } }, [
          React.createElement('li', { key: 'tip-1' }, 'Block size should ideally be a multiple of warp size (32)'),
          React.createElement('li', { key: 'tip-2' }, 'For memory-bound kernels, aim for 128-192 threads per block'),
          React.createElement('li', { key: 'tip-3' }, 'For compute-bound kernels, aim for 256-512 threads per block'),
          React.createElement('li', { key: 'tip-4' }, 'Grid size should ideally be a multiple of block size'),
          React.createElement('li', { key: 'tip-5' }, 'Consider SM resource limits (registers, shared memory)')
        ])
      ])
    ]);
  };

  // Render the component
  console.log('Rendering CUDA visualizer component');
  ReactDOM.render(React.createElement(CUDAVisualization), container);
});
