# Technical Documentation: GPU-Accelerated Image Filtering

This document details the design considerations, algorithmic choices, and optimization strategies implemented in the OpenCL-based GPU image filtering pipeline, specifically focusing on the Median Filter.

## 1. Algorithmic Optimizations: Hybrid Median Search

Calculating the median requires finding the middle value of a local neighborhood. As the filter radius R increases, the window size grows quadratically: N = (2R+1)^2. We implemented a hybrid algorithmic approach (`getMedianAuto`) to maximize performance across different radii:

- **Bubble Sort (Radius <= 4)**: 
  For small windows (e.g., 3x3 to 9x9), the kernel uses a classic Bubble Sort. While its theoretical complexity is O(N^2), it is extremely fast for small N. The fixed, small array size allows the compiler to heavily unroll the loops and keep the data entirely in **fast private registers**, avoiding the overhead of dynamic indexing.
  
- **Histogram-Based Search (Radius > 4)**:
  For larger windows (up to R=15, i.e., 31x31 = 961 pixels), O(N^2) sorting becomes a massive computational bottleneck. The kernel dynamically switches to a Histogram-based median search, which operates in **O(N) time**. It builds a 256-bin frequency map of pixel intensities and linearly finds the median threshold, entirely bypassing the need to sort the elements.

## 2. Memory Hierarchy: Shared (Local) Memory Optimization

The most significant performance limitation for sliding-window filters on GPUs is **Global Memory Bandwidth**.

- **The Naive Approach**: 
  In a naive implementation, every thread independently fetches N pixels from global memory. Because adjacent threads process highly overlapping neighborhoods, the exact same pixels are fetched repeatedly from the slow global VRAM, wasting bandwidth.
  
- **Cooperative Fetching (Tiling + Halo)**: 
  To resolve this, we implemented a **Shared (Local) Memory** strategy. A workgroup (e.g., 16x16 threads) is assigned a specific tile of the image. Before any computation begins, the threads work cooperatively to fetch the target tile **plus the required border pixels (halo/apron)** into `__local` memory. 
  
- **Memory Coalescing**: 
  The cooperative fetch is designed as a linear stride loop (`for (int i = tid; i < total_elements; i += total_threads)`). This guarantees that global memory reads are efficient, maximizing memory bus utilization.
  
- **Barrier Synchronization**: 
  A `barrier(CLK_LOCAL_MEM_FENCE)` ensures all threads have finished fetching the data before the computation phase begins. During the heavy processing phase, threads read their sliding windows exclusively from the ultra-fast, low-latency local memory.

## 3. Boundary Handling (Clamping)

To handle pixels near the edges of the image (where the sliding window would read out of bounds), we utilized the OpenCL built-in `clamp()` function. 
- During the memory fetch phase, if a thread calculates an out-of-bounds global coordinate, it is clamped to the nearest valid edge pixel index.
- This ensures memory safety without relying on complex, highly divergent if/else branching. Eliminating branch divergence during the critical loops keeps the GPU wavefronts/warps synchronized and execution highly efficient.

## 4. Separable Filtering (Gaussian Blur context)

While the median filter is a non-linear operation and cannot be mathematically separated, the project architecture also supports Gaussian blur, which exploits **separable convolution**. 
Instead of a 2D convolution of O(N^2) complexity per pixel, the Gaussian blur is split into two 1D passes (Horizontal, then Vertical), reducing the complexity to O(2N). Paired with similar shared memory optimizations, this results in orders of magnitude faster execution compared to the naive 2D approach.

## 5. Host-Side Architecture

The C++ host application (`main.cpp`) was designed to provide a robust testing and benchmarking environment:
- **Interactive Event Loop**: Allows real-time toggling between CPU, Naive GPU, and Shared GPU implementations, as well as dynamic, on-the-fly radius adjustments via keyboard inputs.
- **High-Resolution Profiling**: Utilizes C++ `std::chrono` for precise micro-benchmarking of the complete API calls and kernel execution times, ensuring accurate performance comparisons between the algorithms.
