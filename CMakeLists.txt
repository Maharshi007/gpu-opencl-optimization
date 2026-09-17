cmake_minimum_required(VERSION 3.20)
project(GPUOpenCLOptimization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(vector_add src/vector_add.cpp)

target_include_directories(vector_add PRIVATE
    "${CMAKE_SOURCE_DIR}/external/OpenCL-Headers/install/include"
)

target_link_libraries(vector_add PRIVATE
    "${CMAKE_SOURCE_DIR}/external/OpenCL-ICD-Loader/build/Release/OpenCL.lib"
)

add_executable(matrix_cpu src/matrix_cpu.cpp)

add_executable(matrix_gpu src/matrix_gpu.cpp)

target_include_directories(matrix_gpu PRIVATE
    "${CMAKE_SOURCE_DIR}/external/OpenCL-Headers/install/include"
)

target_link_libraries(matrix_gpu PRIVATE
    "${CMAKE_SOURCE_DIR}/external/OpenCL-ICD-Loader/build/Release/OpenCL.lib"
)

add_executable(matrix_gpu_tiled src/matrix_gpu_tiled.cpp)

target_include_directories(matrix_gpu_tiled PRIVATE
    "${CMAKE_SOURCE_DIR}/external/OpenCL-Headers/install/include"
)

target_link_libraries(matrix_gpu_tiled PRIVATE
    "${CMAKE_SOURCE_DIR}/external/OpenCL-ICD-Loader/build/Release/OpenCL.lib"
)

add_executable(device_info src/device_info.cpp)

target_include_directories(device_info PRIVATE
    "${CMAKE_SOURCE_DIR}/external/OpenCL-Headers/install/include"
)

target_link_libraries(device_info PRIVATE
    "${CMAKE_SOURCE_DIR}/external/OpenCL-ICD-Loader/build/Release/OpenCL.lib"
)
