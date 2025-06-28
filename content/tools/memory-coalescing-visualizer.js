document.addEventListener('DOMContentLoaded', () => {
  const container = document.getElementById('memory-coalescing-visualization');
  if (!container) {
    console.log('Memory coalescing visualizer container not found');
    return;
  }

  // Create the visualization content
  const content = `
    <div style="
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: #f5f5f5;
      padding: 20px;
      border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      max-width: 100%;
      overflow-x: hidden;
      box-sizing: border-box;
    ">
      <h2 style="color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">GPU Memory Access Patterns</h2>
      
      <div style="margin-bottom: 30px; padding: 15px; border: 2px solid #e0e0e0; border-radius: 8px;">
        <h3 style="color: #333;">Global Memory Coalescing</h3>
        <p><strong>Key Principle:</strong> When threads in a warp access consecutive memory addresses, the hardware combines these into fewer transactions.</p>
        
        <div style="display: grid; grid-template-columns: 1fr; gap: 20px; margin: 20px 0;">
          <div style="border-left: 5px solid #4CAF50; padding: 15px;">
            <h4>Coalesced Access (Good)</h4>
            <p>Threads access consecutive elements</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(40px, 1fr)); gap: 2px; margin: 15px 0; justify-content: center;">
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #ff6b6b; border: 3px solid #4CAF50;">T0→0</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #4ecdc4; border: 3px solid #4CAF50;">T1→1</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #45b7d1; border: 3px solid #4CAF50;">T2→2</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #96ceb4; border: 3px solid #4CAF50;">T3→3</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #feca57; border: 3px solid #4CAF50;">T4→4</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #ff9ff3; border: 3px solid #4CAF50;">T5→5</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #54a0ff; border: 3px solid #4CAF50;">T6→6</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #5f27cd; border: 3px solid #4CAF50;">T7→7</div>
            </div>
            <p><strong>Result:</strong> 1 memory transaction for 8 elements</p>
          </div>
          
          <div style="border-left: 5px solid #f44336; padding: 15px;">
            <h4>Strided Access (Bad)</h4>
            <p>Threads access with stride = 8</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(40px, 1fr)); gap: 2px; margin: 15px 0; justify-content: center;">
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #ff6b6b; border: 3px solid #f44336;">T0→0</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold;">1</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold;">2</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold;">3</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold;">4</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold;">5</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold;">6</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold;">7</div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(40px, 1fr)); gap: 2px; margin: 15px 0; justify-content: center;">
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #4ecdc4; border: 3px solid #f44336;">T1→8</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold;">9</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold;">10</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold;">11</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold;">12</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold;">13</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold;">14</div>
              <div style="width: 40px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold;">15</div>
            </div>
            <p><strong>Result:</strong> 8 separate memory transactions</p>
          </div>
        </div>
        
        <div style="background: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 5px; font-family: 'Courier New', monospace; margin: 10px 0; overflow-x: auto; max-width: 100%; font-size: 12px;">
          <div>// Good: Coalesced access pattern</div>
          <div>for (int i = threadIdx.x; i < N; i += blockDim.x) {</div>
          <div>    data[i] = compute(i);  // Each thread accesses consecutive elements</div>
          <div>}</div>
          <div></div>
          <div>// Bad: Strided access pattern</div>
          <div>for (int i = 0; i < N; i++) {</div>
          <div>    data[i * blockDim.x + threadIdx.x] = compute(i);  // Large strides</div>
          <div>}</div>
        </div>
      </div>
      
      <div style="margin-bottom: 30px; padding: 15px; border: 2px solid #e0e0e0; border-radius: 8px;">
        <h3 style="color: #333;">Shared Memory Bank Conflicts</h3>
        <p><strong>Shared memory is divided into 32 banks</strong>. Bank conflicts occur when multiple threads access different addresses in the same bank simultaneously.</p>
        
        <h4>Memory Banking Layout (32 banks, 4-byte words)</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(60px, 1fr)); gap: 2px; margin: 15px 0; justify-content: center;">
          <div style="width: 60px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #ffebee;">Bank 0<br>Addr 0</div>
          <div style="width: 60px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #e8f5e8;">Bank 1<br>Addr 4</div>
          <div style="width: 60px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #e3f2fd;">Bank 2<br>Addr 8</div>
          <div style="width: 60px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #fff3e0;">Bank 3<br>Addr 12</div>
          <div style="width: 60px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #ffebee;">Bank 4<br>Addr 16</div>
          <div style="width: 60px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #e8f5e8;">Bank 5<br>Addr 20</div>
          <div style="width: 60px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #e3f2fd;">Bank 6<br>Addr 24</div>
          <div style="width: 60px; height: 40px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; background-color: #fff3e0;">Bank 7<br>Addr 28</div>
        </div>
        
        <div style="background: #dbeafe; padding: 16px; border-radius: 4px; font-size: 14px; margin-top: 20px;">
          <p style="font-weight: bold; margin: 0 0 8px 0;">Key Takeaways:</p>
          <ul style="list-style-type: disc; padding-left: 20px; margin: 8px 0;">
            <li>Coalesced memory access patterns significantly improve performance</li>
            <li>Strided access patterns should be avoided when possible</li>
            <li>Bank conflicts in shared memory can be mitigated through proper data layout</li>
            <li>Understanding memory hierarchy is crucial for GPU optimization</li>
          </ul>
        </div>
      </div>
    </div>
  `;

  container.innerHTML = content;
});
