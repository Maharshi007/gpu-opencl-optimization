#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
    constexpr int N = 1024;

    constexpr int WARMUP_RUNS = 1;
    constexpr int BENCHMARK_RUNS = 20;

    // Current experiment configuration.
    constexpr size_t WORK_GROUP_SIZE = 8;

    const size_t matrix_elements =
        static_cast<size_t>(N) * N;

    const size_t matrix_bytes =
        matrix_elements * sizeof(float);

    // ------------------------------------------------------------
    // 1. Allocate matrices.
    // ------------------------------------------------------------
    std::vector<float> A(matrix_elements);
    std::vector<float> B(matrix_elements);
    std::vector<float> C(matrix_elements, 0.0f);

    // Same deterministic input used by the CPU
    // and tiled GPU versions.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] =
                static_cast<float>((i + j) % 100);

            B[i * N + j] =
                static_cast<float>((i - j + 100) % 100);
        }
    }

    cl_int err = CL_SUCCESS;

    // ------------------------------------------------------------
    // 2. Find OpenCL platforms.
    // ------------------------------------------------------------
    cl_uint num_platforms = 0;

    err = clGetPlatformIDs(
        0,
        nullptr,
        &num_platforms
    );

    if (err != CL_SUCCESS || num_platforms == 0) {
        std::cerr
            << "Error: No OpenCL platforms found.\n";
        return 1;
    }

    std::vector<cl_platform_id> platforms(
        num_platforms
    );

    err = clGetPlatformIDs(
        num_platforms,
        platforms.data(),
        nullptr
    );

    if (err != CL_SUCCESS) {
        std::cerr
            << "Error: Could not get OpenCL platforms.\n";
        return 1;
    }

    cl_platform_id selected_platform = nullptr;

    for (cl_platform_id platform : platforms) {
        size_t name_size = 0;

        clGetPlatformInfo(
            platform,
            CL_PLATFORM_NAME,
            0,
            nullptr,
            &name_size
        );

        std::string name(
            name_size,
            '\0'
        );

        clGetPlatformInfo(
            platform,
            CL_PLATFORM_NAME,
            name_size,
            name.data(),
            nullptr
        );

        std::cout
            << "Found platform: "
            << name.c_str()
            << '\n';

        if (
            name.find("NVIDIA") != std::string::npos ||
            name.find("CUDA") != std::string::npos
        ) {
            selected_platform = platform;
        }
    }

    if (selected_platform == nullptr) {
        std::cerr
            << "Error: NVIDIA OpenCL platform not found.\n";
        return 1;
    }

    // ------------------------------------------------------------
    // 3. Select NVIDIA GPU.
    // ------------------------------------------------------------
    cl_device_id device = nullptr;

    err = clGetDeviceIDs(
        selected_platform,
        CL_DEVICE_TYPE_GPU,
        1,
        &device,
        nullptr
    );

    if (err != CL_SUCCESS) {
        std::cerr
            << "Error: Could not find NVIDIA GPU device.\n";
        return 1;
    }

    char device_name[256] = {};

    clGetDeviceInfo(
        device,
        CL_DEVICE_NAME,
        sizeof(device_name),
        device_name,
        nullptr
    );

    std::cout
        << "Using device: "
        << device_name
        << '\n';

    // ------------------------------------------------------------
    // 4. Create OpenCL context.
    // ------------------------------------------------------------
    cl_context context =
        clCreateContext(
            nullptr,
            1,
            &device,
            nullptr,
            nullptr,
            &err
        );

    if (err != CL_SUCCESS || context == nullptr) {
        std::cerr
            << "Error: Could not create OpenCL context.\n";
        return 1;
    }

    // ------------------------------------------------------------
    // 5. Create command queue with profiling enabled.
    // ------------------------------------------------------------
    cl_command_queue queue =
        clCreateCommandQueue(
            context,
            device,
            CL_QUEUE_PROFILING_ENABLE,
            &err
        );

    if (err != CL_SUCCESS || queue == nullptr) {
        std::cerr
            << "Error: Could not create command queue.\n";

        clReleaseContext(context);

        return 1;
    }

    // ------------------------------------------------------------
    // 6. Load the naive OpenCL kernel.
    // ------------------------------------------------------------
    std::ifstream kernel_file(
        "kernels/matrix_mul.cl"
    );

    if (!kernel_file) {
        std::cerr
            << "Error: Could not open matrix_mul.cl.\n";

        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return 1;
    }

    std::stringstream kernel_stream;

    kernel_stream
        << kernel_file.rdbuf();

    std::string kernel_source =
        kernel_stream.str();

    const char* source =
        kernel_source.c_str();

    const size_t source_size =
        kernel_source.size();

    // ------------------------------------------------------------
    // 7. Create and build OpenCL program.
    // ------------------------------------------------------------
    cl_program program =
        clCreateProgramWithSource(
            context,
            1,
            &source,
            &source_size,
            &err
        );

    if (err != CL_SUCCESS) {
        std::cerr
            << "Error: Could not create OpenCL program.\n";

        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return 1;
    }

    err = clBuildProgram(
        program,
        1,
        &device,
        nullptr,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS) {
        std::cerr
            << "Error: Could not build naive kernel.\n";

        size_t log_size = 0;

        clGetProgramBuildInfo(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            0,
            nullptr,
            &log_size
        );

        std::string build_log(
            log_size,
            '\0'
        );

        clGetProgramBuildInfo(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            log_size,
            build_log.data(),
            nullptr
        );

        std::cerr
            << build_log
            << '\n';

        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return 1;
    }

    // ------------------------------------------------------------
    // 8. Create kernel object.
    // ------------------------------------------------------------
    cl_kernel kernel =
        clCreateKernel(
            program,
            "matrix_multiply",
            &err
        );

    if (err != CL_SUCCESS) {
        std::cerr
            << "Error: Could not create matrix_multiply kernel.\n";

        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return 1;
    }

    // ------------------------------------------------------------
    // 9. Create GPU buffers.
    // ------------------------------------------------------------
    cl_mem buffer_A =
        clCreateBuffer(
            context,
            CL_MEM_READ_ONLY,
            matrix_bytes,
            nullptr,
            &err
        );

    if (err != CL_SUCCESS) {
        std::cerr
            << "Error: Could not create buffer A.\n";
        return 1;
    }

    cl_mem buffer_B =
        clCreateBuffer(
            context,
            CL_MEM_READ_ONLY,
            matrix_bytes,
            nullptr,
            &err
        );

    if (err != CL_SUCCESS) {
        std::cerr
            << "Error: Could not create buffer B.\n";
        return 1;
    }

    cl_mem buffer_C =
        clCreateBuffer(
            context,
            CL_MEM_WRITE_ONLY,
            matrix_bytes,
            nullptr,
            &err
        );

    if (err != CL_SUCCESS) {
        std::cerr
            << "Error: Could not create buffer C.\n";
        return 1;
    }

    // ------------------------------------------------------------
    // 10. Copy matrices to GPU.
    // ------------------------------------------------------------
    err = clEnqueueWriteBuffer(
        queue,
        buffer_A,
        CL_TRUE,
        0,
        matrix_bytes,
        A.data(),
        0,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS) {
        std::cerr
            << "Error: Could not copy A to GPU.\n";
        return 1;
    }

    err = clEnqueueWriteBuffer(
        queue,
        buffer_B,
        CL_TRUE,
        0,
        matrix_bytes,
        B.data(),
        0,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS) {
        std::cerr
            << "Error: Could not copy B to GPU.\n";
        return 1;
    }

    // ------------------------------------------------------------
    // 11. Set kernel arguments.
    // ------------------------------------------------------------
    err = clSetKernelArg(
        kernel,
        0,
        sizeof(cl_mem),
        &buffer_A
    );

    if (err != CL_SUCCESS) {
        std::cerr
            << "Error: Could not set kernel argument A.\n";
        return 1;
    }

    err = clSetKernelArg(
        kernel,
        1,
        sizeof(cl_mem),
        &buffer_B
    );

    if (err != CL_SUCCESS) {
        std::cerr
            << "Error: Could not set kernel argument B.\n";
        return 1;
    }

    err = clSetKernelArg(
        kernel,
        2,
        sizeof(cl_mem),
        &buffer_C
    );

    if (err != CL_SUCCESS) {
        std::cerr
            << "Error: Could not set kernel argument C.\n";
        return 1;
    }

    err = clSetKernelArg(
        kernel,
        3,
        sizeof(int),
        &N
    );

    if (err != CL_SUCCESS) {
        std::cerr
            << "Error: Could not set kernel argument N.\n";
        return 1;
    }

    // ------------------------------------------------------------
    // 12. Define global and local work sizes.
    // ------------------------------------------------------------
    const size_t global_work_size[2] = {
        static_cast<size_t>(N),
        static_cast<size_t>(N)
    };

    const size_t local_work_size[2] = {
        WORK_GROUP_SIZE,
        WORK_GROUP_SIZE
    };

    // ------------------------------------------------------------
    // 13. Warm-up run.
    // ------------------------------------------------------------
    for (int run = 0;
         run < WARMUP_RUNS;
         ++run) {

        err = clEnqueueNDRangeKernel(
            queue,
            kernel,
            2,
            nullptr,
            global_work_size,
            local_work_size,
            0,
            nullptr,
            nullptr
        );

        if (err != CL_SUCCESS) {
            std::cerr
                << "Error: Could not launch warm-up kernel.\n";
            return 1;
        }
    }

    clFinish(queue);

    // ------------------------------------------------------------
    // 14. Repeated benchmark.
    // ------------------------------------------------------------
    std::vector<double> kernel_times_ms;

    kernel_times_ms.reserve(
        BENCHMARK_RUNS
    );

    for (int run = 0;
         run < BENCHMARK_RUNS;
         ++run) {

        cl_event kernel_event = nullptr;

        err = clEnqueueNDRangeKernel(
            queue,
            kernel,
            2,
            nullptr,
            global_work_size,
            local_work_size,
            0,
            nullptr,
            &kernel_event
        );

        if (err != CL_SUCCESS) {
            std::cerr
                << "Error: Could not launch benchmark kernel.\n";
            return 1;
        }

        clFinish(queue);

        cl_ulong start_time = 0;
        cl_ulong end_time = 0;

        err = clGetEventProfilingInfo(
            kernel_event,
            CL_PROFILING_COMMAND_START,
            sizeof(cl_ulong),
            &start_time,
            nullptr
        );

        if (err != CL_SUCCESS) {
            std::cerr
                << "Error: Could not get kernel start time.\n";

            clReleaseEvent(kernel_event);

            return 1;
        }

        err = clGetEventProfilingInfo(
            kernel_event,
            CL_PROFILING_COMMAND_END,
            sizeof(cl_ulong),
            &end_time,
            nullptr
        );

        if (err != CL_SUCCESS) {
            std::cerr
                << "Error: Could not get kernel end time.\n";

            clReleaseEvent(kernel_event);

            return 1;
        }

        const double kernel_time_ms =
            static_cast<double>(
                end_time - start_time
            ) / 1'000'000.0;

        kernel_times_ms.push_back(
            kernel_time_ms
        );

        clReleaseEvent(kernel_event);
    }

    // ------------------------------------------------------------
    // 15. Calculate statistics.
    // ------------------------------------------------------------
    double total_time_ms = 0.0;

    double min_time_ms =
        kernel_times_ms[0];

    double max_time_ms =
        kernel_times_ms[0];

    for (double time_ms : kernel_times_ms) {
        total_time_ms += time_ms;

        min_time_ms =
            std::min(
                min_time_ms,
                time_ms
            );

        max_time_ms =
            std::max(
                max_time_ms,
                time_ms
            );
    }

    const double average_time_ms =
        total_time_ms /
        static_cast<double>(
            BENCHMARK_RUNS
        );

    std::vector<double> sorted_times =
        kernel_times_ms;

    std::sort(
        sorted_times.begin(),
        sorted_times.end()
    );

    double median_time_ms = 0.0;

    if (BENCHMARK_RUNS % 2 == 0) {
        median_time_ms =
            (
                sorted_times[
                    BENCHMARK_RUNS / 2 - 1
                ] +
                sorted_times[
                    BENCHMARK_RUNS / 2
                ]
            ) / 2.0;
    } else {
        median_time_ms =
            sorted_times[
                BENCHMARK_RUNS / 2
            ];
    }

    // ------------------------------------------------------------
    // 16. Copy result from GPU to CPU.
    // ------------------------------------------------------------
    err = clEnqueueReadBuffer(
        queue,
        buffer_C,
        CL_TRUE,
        0,
        matrix_bytes,
        C.data(),
        0,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS) {
        std::cerr
            << "Error: Could not read result from GPU.\n";
        return 1;
    }

    // ------------------------------------------------------------
    // 17. Calculate checksum.
    // ------------------------------------------------------------
    double checksum = 0.0;

    for (float value : C) {
        checksum += value;
    }

    // Verified independently using the 1024x1024 CPU baseline.
    const double expected_checksum =
        488745369600.0;

    const double tolerance =
        100000.0;

    // ------------------------------------------------------------
    // 18. Print benchmark results.
    // ------------------------------------------------------------
    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "\n========== NAIVE GPU BENCHMARK ==========\n";

    std::cout
        << "Matrix size: "
        << N
        << " x "
        << N
        << '\n';

    std::cout
        << "Work-group size: "
        << WORK_GROUP_SIZE
        << " x "
        << WORK_GROUP_SIZE
        << '\n';

    std::cout
        << "Benchmark runs: "
        << BENCHMARK_RUNS
        << '\n';

    std::cout
        << "Average naive kernel time: "
        << average_time_ms
        << " ms\n";

    std::cout
        << "Median naive kernel time: "
        << median_time_ms
        << " ms\n";

    std::cout
        << "Minimum naive kernel time: "
        << min_time_ms
        << " ms\n";

    std::cout
        << "Maximum naive kernel time: "
        << max_time_ms
        << " ms\n";

    std::cout
        << "Checksum: "
        << checksum
        << '\n';

    // ------------------------------------------------------------
    // 19. Validation result.
    // ------------------------------------------------------------
    if (
        std::abs(
            checksum - expected_checksum
        ) <= tolerance
    ) {
        std::cout
            << "Naive GPU result validation successful!\n";
    } else {
        std::cout
            << "Warning: Naive GPU checksum differs "
            << "from CPU baseline.\n";
    }

    std::cout
        << "=========================================\n";

    // ------------------------------------------------------------
    // 20. Release OpenCL resources.
    // ------------------------------------------------------------
    clReleaseMemObject(buffer_A);
    clReleaseMemObject(buffer_B);
    clReleaseMemObject(buffer_C);

    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}