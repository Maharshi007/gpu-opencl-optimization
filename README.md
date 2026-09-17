\# GPU/OpenCL Kernel Performance Optimization



A C++ and OpenCL project focused on understanding and optimizing GPU kernel performance through parallel execution, work-group configuration, memory reuse, and benchmarking.



\## Overview



This project implements matrix multiplication using three execution approaches:



1\. CPU baseline using C++

2\. Naive GPU implementation using OpenCL

3\. Tiled GPU implementation using OpenCL local memory



The project evaluates how work-group size and local-memory tiling affect kernel execution time on an NVIDIA GeForce GTX 1650.



The main objective is to understand practical GPU performance optimization concepts such as:



\- GPU parallel execution

\- OpenCL kernels

\- Global and local work sizes

\- Work-group configuration

\- Memory access patterns

\- Local memory and data reuse

\- Kernel profiling

\- Performance bottlenecks

\- Benchmarking and validation



\---



\## Project Structure



```text

gpu-opencl-optimization/

│

├── kernels/

│   ├── vector\_add.cl

│   ├── matrix\_mul.cl

│   └── matrix\_mul\_tiled.cl

│

├── src/

│   ├── vector\_add.cpp

│   ├── device\_info.cpp

│   ├── matrix\_cpu.cpp

│   ├── matrix\_gpu.cpp

│   └── matrix\_gpu\_tiled.cpp

│

├── results/

│   └── benchmark.csv

│

├── CMakeLists.txt

├── README.md

└── .gitignore

