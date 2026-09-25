# GPU/OpenCL Kernel Performance Optimization

A C++ and OpenCL project focused on understanding and optimizing GPU kernel performance through parallel execution, work-group configuration, memory reuse, and benchmarking.

---

## Overview

This project implements matrix multiplication using three execution approaches:

1. CPU baseline using C++
2. Naive GPU implementation using OpenCL
3. Tiled GPU implementation using OpenCL local memory

The project evaluates how work-group size and local-memory tiling affect kernel execution time on an NVIDIA GeForce GTX 1650.

The main objective is to understand practical GPU performance optimization concepts such as:

- GPU parallel execution
- OpenCL kernels
- Global and local work sizes
- Work-group configuration
- Memory access patterns
- Local memory and data reuse
- Kernel profiling
- Performance bottlenecks
- Benchmarking and validation

---

## Objectives

The project focuses on the following goals:

- Establish a CPU matrix multiplication baseline.
- Implement matrix multiplication as an OpenCL GPU kernel.
- Compare a naive GPU implementation against a tiled implementation.
- Study the effect of different work-group and tile sizes.
- Use OpenCL local memory to improve data reuse.
- Measure kernel execution time using OpenCL event profiling.
- Validate GPU results against the expected output.
- Analyze performance using numerical results and graphs.

---

## Technologies

- **C++**
- **OpenCL**
- **CMake**
- **Python**
- **Matplotlib**
- **Git**
- **NVIDIA GeForce GTX 1650**
- **Windows**
- **Visual Studio Build Tools**

---

## Project Structure

```text
gpu-opencl-optimization/
│
├── kernels/
│   ├── vector_add.cl
│   ├── matrix_mul.cl
│   └── matrix_mul_tiled.cl
│
├── src/
│   ├── vector_add.cpp
│   ├── device_info.cpp
│   ├── matrix_cpu.cpp
│   ├── matrix_gpu.cpp
│   └── matrix_gpu_tiled.cpp
│
├── results/
│   ├── final_results.csv
│   ├── benchmark.csv
│   ├── analyze_results.py
│   └── figures/
│       ├── overall_performance.png
│       └── tile_size_sweep.png
│
├── CMakeLists.txt
├── README.md
└── .gitignore