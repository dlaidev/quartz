import React, { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';

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

  const renderBlocks = () => {
    const blocks = [];
    
    for (let b = 0; b < numBlocks; b++) {
      const isLastBlock = b === numBlocks - 1;
      const currentBlockSize = isLastBlock ? lastBlockSize : blockSize;
      const currentWarps = isLastBlock ? lastBlockWarps : warpsPerBlock;
      
      const blockStyle = {
        display: 'flex',
        flexDirection: 'column',
        border: '2px solid #334155',
        borderRadius: '4px',
        margin: '4px',
        backgroundColor: '#1e293b',
        width: `${Math.max(100, (currentBlockSize / blockSize) * 200)}px`,
        overflow: 'hidden'
      };
      
      const warps = [];
      for (let w = 0; w < currentWarps; w++) {
        const isLastWarp = isLastBlock && w === currentWarps - 1 && lastWarpSize < warpSize;
        const currentWarpSize = isLastWarp ? lastWarpSize : Math.min(warpSize, currentBlockSize - w * warpSize);

        const warpStyle = {
          display: 'flex',
          flexWrap: 'wrap',
          padding: '2px',
          margin: '2px',
          backgroundColor: '#0f172a',
          borderRadius: '3px',
          position: 'relative'
        };
        
        const threads = [];
        for (let t = 0; t < warpSize; t++) {
          const isActiveThread = t < currentWarpSize;
          const threadStyle = {
            width: '8px',
            height: '8px',
            margin: '1px',
            borderRadius: '50%',
            backgroundColor: isActiveThread ? '#22c55e' : '#64748b',
            opacity: isActiveThread ? 1 : 0.3
          };
          
          threads.push(<div key={t} style={threadStyle} />);
        }
        
        warps.push(
          <div key={w} style={warpStyle} title={`Warp ${b}-${w}: ${currentWarpSize}/${warpSize} threads`}>
            <div style={{ display: 'flex', flexWrap: 'wrap' }}>
              {threads}
            </div>
          </div>
        );
      }
      
      blocks.push(
        <div key={b} style={blockStyle} title={`Block ${b}: ${currentBlockSize} threads`}>
          <div style={{ backgroundColor: '#334155', padding: '2px 6px', textAlign: 'center', fontSize: '12px' }}>
            Block {b}
          </div>
          <div>{warps}</div>
        </div>
      );
    }
    
    return blocks;
  };

  return (
    <div className="p-4 bg-gray-100 rounded-lg">
      <h2 className="text-xl font-bold mb-4">Interactive CUDA Thread/Block Visualizer</h2>
      
      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <label className="font-medium">Grid Size: {gridSize} threads</label>
          <input 
            type="range"
            min="1" 
            max="512" 
            value={gridSize} 
            onChange={(e) => setGridSize(parseInt(e.target.value))}
            className="w-48"
          />
        </div>
        
        <div className="flex justify-between items-center">
          <label className="font-medium">Block Size: {blockSize} threads</label>
          <input 
            type="range"
            min="1" 
            max="256" 
            value={blockSize} 
            onChange={(e) => setBlockSize(parseInt(e.target.value))}
            className="w-48"
          />
        </div>
      </div>
      
      <div className="mb-4 bg-white p-4 rounded shadow-sm">
        <div className="flex flex-wrap">{renderBlocks()}</div>
      </div>
      
      <div className="bg-white p-4 rounded shadow-sm mb-4">
        <div className="flex justify-between items-center cursor-pointer" onClick={() => setShowDetails(!showDetails)}>
          <h3 className="font-bold">Configuration Analysis</h3>
          {showDetails ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </div>
        
        {showDetails && (
          <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
            <div className="bg-gray-50 p-2 rounded">
              <p><strong>Blocks:</strong> {numBlocks}</p>
              <p><strong>Block Efficiency:</strong> {blockEfficiency.toFixed(1)}%</p>
              <p><strong>Last Block Size:</strong> {lastBlockSize} threads</p>
              <p><strong>Last Block Efficiency:</strong> {lastBlockEfficiency.toFixed(1)}%</p>
            </div>
            <div className="bg-gray-50 p-2 rounded">
              <p><strong>Warps per Block:</strong> {warpsPerBlock}</p>
              <p><strong>Total Warps:</strong> {totalWarps}</p>
              <p><strong>Last Warp Size:</strong> {lastWarpSize} threads</p>
              <p><strong>Warp Utilization:</strong> {warpUtilization.toFixed(1)}%</p>
            </div>
            <div className="col-span-2 bg-gray-50 p-2 rounded mt-2">
              <p><strong>Memory Coalescing:</strong> {
                blockSize % 32 === 0 ? 
                "Optimal (block size is a multiple of warp size)" : 
                "Sub-optimal (block size is not a multiple of warp size)"
              }</p>
              <p><strong>Occupancy Impact:</strong> {
                blockSize <= 128 ? "Good (smaller blocks allow more concurrent blocks)" :
                blockSize <= 256 ? "Moderate (medium blocks balance resources)" :
                "Low (large blocks may limit concurrency)"
              }</p>
              <p><strong>Thread Divergence Risk:</strong> {
                gridSize % blockSize === 0 ? "Low (even distribution of threads)" :
                lastBlockSize < blockSize / 2 ? "High (last block significantly underutilized)" :
                "Moderate (some underutilization in last block)"
              }</p>
            </div>
          </div>
        )}
      </div>
      
      <div className="bg-blue-50 p-4 rounded text-sm">
        <p><strong>Tips for Optimal Configuration:</strong></p>
        <ul className="list-disc pl-5 mt-2">
          <li>Block size should ideally be a multiple of warp size (32)</li>
          <li>For memory-bound kernels, aim for 128-192 threads per block</li>
          <li>For compute-bound kernels, aim for 256-512 threads per block</li>
          <li>Grid size should ideally be a multiple of block size</li>
          <li>Consider SM resource limits (registers, shared memory)</li>
        </ul>
      </div>
    </div>
  );
};

export default CUDAVisualization;