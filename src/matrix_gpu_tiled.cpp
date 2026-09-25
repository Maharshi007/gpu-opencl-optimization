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

constexpr int TILE_SIZE = 8;
constexpr int MATRIX_SIZE = 1024;
constexpr int WARMUP_RUNS = 1;
constexpr int BENCHMARK_RUNS = 20;

struct BenchmarkResult {
    int tile_size = 0;

    double average_ms = 0.0;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;

    double checksum = 0.0;

    bool validation_passed = false;
    bool execution_success = false;
};


// ============================================================
// Run one complete tiled OpenCL benchmark.
// ============================================================
BenchmarkResult run_benchmark(int tile_size) {

    BenchmarkResult result;
    result.tile_size = tile_size;

    constexpr int N = MATRIX_SIZE;

    // --------------------------------------------------------
    // 1. Validate tile size.
    // --------------------------------------------------------
    if (tile_size <= 0) {
        std::cerr
            << "Error: Tile size must be positive.\n";

        return result;
    }

    if (N % tile_size != 0) {
        std::cerr
            << "Error: Tile size "
            << tile_size
            << " does not evenly divide matrix size "
            << N
            << ".\n";

        return result;
    }

    const size_t work_items_per_group =
        static_cast<size_t>(tile_size) *
        static_cast<size_t>(tile_size);

    if (work_items_per_group > 256) {
        std::cerr
            << "Error: Tile size "
            << tile_size
            << " creates "
            << work_items_per_group
            << " work-items per work-group.\n"
            << "Maximum supported work-group size for this "
            << "benchmark is 256.\n";

        return result;
    }

    // --------------------------------------------------------
    // 2. Allocate matrices.
    // --------------------------------------------------------
    const size_t matrix_elements =
        static_cast<size_t>(N) *
        static_cast<size_t>(N);

    const size_t matrix_bytes =
        matrix_elements *
        sizeof(float);

    std::vector<float> A(matrix_elements);
    std::vector<float> B(matrix_elements);
    std::vector<float> C(matrix_elements, 0.0f);

    // Same deterministic input used by the CPU and naive GPU
    // baselines.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {

            A[i * N + j] =
                static_cast<float>(
                    (i + j) % 100
                );

            B[i * N + j] =
                static_cast<float>(
                    (i - j + 100) % 100
                );
        }
    }

    cl_int err = CL_SUCCESS;

    // --------------------------------------------------------
    // 3. Find OpenCL platforms.
    // --------------------------------------------------------
    cl_uint num_platforms = 0;

    err = clGetPlatformIDs(
        0,
        nullptr,
        &num_platforms
    );

    if (err != CL_SUCCESS || num_platforms == 0) {

        std::cerr
            << "Error: No OpenCL platforms found.\n";

        return result;
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

        return result;
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

        return result;
    }

    // --------------------------------------------------------
    // 4. Select NVIDIA GPU device.
    // --------------------------------------------------------
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

        return result;
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

    // --------------------------------------------------------
    // 5. Query maximum work-group size.
    // --------------------------------------------------------
    size_t max_work_group_size = 0;

    err = clGetDeviceInfo(
        device,
        CL_DEVICE_MAX_WORK_GROUP_SIZE,
        sizeof(size_t),
        &max_work_group_size,
        nullptr
    );

    if (err != CL_SUCCESS) {

        std::cerr
            << "Error: Could not query maximum work-group size.\n";

        return result;
    }

    if (work_items_per_group > max_work_group_size) {

        std::cerr
            << "Error: Tile size "
            << tile_size
            << " requires "
            << work_items_per_group
            << " work-items, but device supports only "
            << max_work_group_size
            << ".\n";

        return result;
    }

    // --------------------------------------------------------
    // 6. Create OpenCL context.
    // --------------------------------------------------------
    cl_context context =
        clCreateContext(
            nullptr,
            1,
            &device,
            nullptr,
            nullptr,
            &err
        );

    if (
        err != CL_SUCCESS ||
        context == nullptr
    ) {

        std::cerr
            << "Error: Could not create OpenCL context.\n";

        return result;
    }

    // --------------------------------------------------------
    // 7. Create command queue with profiling enabled.
    // --------------------------------------------------------
    cl_command_queue queue =
        clCreateCommandQueue(
            context,
            device,
            CL_QUEUE_PROFILING_ENABLE,
            &err
        );

    if (
        err != CL_SUCCESS ||
        queue == nullptr
    ) {

        std::cerr
            << "Error: Could not create command queue.\n";

        clReleaseContext(context);

        return result;
    }

    // --------------------------------------------------------
    // 8. Load tiled OpenCL kernel.
    // --------------------------------------------------------
    std::ifstream kernel_file(
        "kernels/matrix_mul_tiled.cl"
    );

    if (!kernel_file) {

        std::cerr
            << "Error: Could not open tiled kernel file.\n";

        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return result;
    }

    std::stringstream kernel_stream;

    kernel_stream
        << kernel_file.rdbuf();

    const std::string kernel_source =
        kernel_stream.str();

    const char* source =
        kernel_source.c_str();

    const size_t source_size =
        kernel_source.size();

    // --------------------------------------------------------
    // 9. Create OpenCL program.
    // --------------------------------------------------------
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

        return result;
    }

    // --------------------------------------------------------
    // 10. Compile kernel with runtime-selected tile size.
    // --------------------------------------------------------
    const std::string build_options =
        "-DTILE_SIZE=" +
        std::to_string(tile_size);

    err = clBuildProgram(
        program,
        1,
        &device,
        build_options.c_str(),
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS) {

        std::cerr
            << "Error: Could not build tiled kernel.\n";

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

        return result;
    }

    // --------------------------------------------------------
    // 11. Create kernel object.
    // --------------------------------------------------------
    cl_kernel kernel =
        clCreateKernel(
            program,
            "matrix_multiply_tiled",
            &err
        );

    if (err != CL_SUCCESS) {

        std::cerr
            << "Error: Could not create tiled kernel.\n";

        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return result;
    }

    // --------------------------------------------------------
    // 12. Create GPU buffers.
    // --------------------------------------------------------
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

        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return result;
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

        clReleaseMemObject(buffer_A);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return result;
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

        clReleaseMemObject(buffer_A);
        clReleaseMemObject(buffer_B);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return result;
    }

    // --------------------------------------------------------
    // 13. Copy input matrices to GPU.
    // --------------------------------------------------------
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
            << "Error: Could not copy matrix A to GPU.\n";

        clReleaseMemObject(buffer_A);
        clReleaseMemObject(buffer_B);
        clReleaseMemObject(buffer_C);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return result;
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
            << "Error: Could not copy matrix B to GPU.\n";

        clReleaseMemObject(buffer_A);
        clReleaseMemObject(buffer_B);
        clReleaseMemObject(buffer_C);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return result;
    }

    // --------------------------------------------------------
    // 14. Set kernel arguments.
    // --------------------------------------------------------
    err = clSetKernelArg(
        kernel,
        0,
        sizeof(cl_mem),
        &buffer_A
    );

    if (err != CL_SUCCESS) {

        std::cerr
            << "Error: Could not set kernel argument A.\n";

        return result;
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

        return result;
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

        return result;
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

        return result;
    }

    // --------------------------------------------------------
    // 15. Define global and local work sizes.
    // --------------------------------------------------------
    const size_t global_work_size[2] = {
        static_cast<size_t>(N),
        static_cast<size_t>(N)
    };

    const size_t local_work_size[2] = {
        static_cast<size_t>(tile_size),
        static_cast<size_t>(tile_size)
    };

    // --------------------------------------------------------
    // 16. Warm-up run.
    // --------------------------------------------------------
    for (
        int run = 0;
        run < WARMUP_RUNS;
        ++run
    ) {

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

            return result;
        }
    }

    clFinish(queue);

    // --------------------------------------------------------
    // 17. Repeated benchmark.
    // --------------------------------------------------------
    std::vector<double> kernel_times_ms;

    kernel_times_ms.reserve(
        BENCHMARK_RUNS
    );

    for (
        int run = 0;
        run < BENCHMARK_RUNS;
        ++run
    ) {

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

            return result;
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

            return result;
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

            return result;
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

    // --------------------------------------------------------
    // 18. Calculate benchmark statistics.
    // --------------------------------------------------------
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

    // --------------------------------------------------------
    // 19. Copy result from GPU to CPU.
    // --------------------------------------------------------
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

        return result;
    }

    // --------------------------------------------------------
    // 20. Calculate checksum.
    // --------------------------------------------------------
    double checksum = 0.0;

    for (float value : C) {
        checksum += value;
    }

    // Independently verified CPU baseline checksum.
    const double expected_checksum =
        488745369600.0;

    const double tolerance =
        100000.0;

    const bool validation_passed =
        std::abs(
            checksum - expected_checksum
        ) <= tolerance;

    // --------------------------------------------------------
    // 21. Store benchmark result.
    // --------------------------------------------------------
    result.average_ms =
        average_time_ms;

    result.median_ms =
        median_time_ms;

    result.min_ms =
        min_time_ms;

    result.max_ms =
        max_time_ms;

    result.checksum =
        checksum;

    result.validation_passed =
        validation_passed;

    result.execution_success =
        true;

    // --------------------------------------------------------
    // 22. Print benchmark results.
    // --------------------------------------------------------
    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "\n========== TILED GPU BENCHMARK ==========\n";

    std::cout
        << "Matrix size: "
        << N
        << " x "
        << N
        << '\n';

    std::cout
        << "Tile size: "
        << tile_size
        << " x "
        << tile_size
        << '\n';

    std::cout
        << "Work-items per work-group: "
        << work_items_per_group
        << '\n';

    std::cout
        << "Benchmark runs: "
        << BENCHMARK_RUNS
        << '\n';

    std::cout
        << "Average tiled kernel time: "
        << average_time_ms
        << " ms\n";

    std::cout
        << "Median tiled kernel time: "
        << median_time_ms
        << " ms\n";

    std::cout
        << "Minimum tiled kernel time: "
        << min_time_ms
        << " ms\n";

    std::cout
        << "Maximum tiled kernel time: "
        << max_time_ms
        << " ms\n";

    std::cout
        << "Checksum: "
        << checksum
        << '\n';

    if (validation_passed) {

        std::cout
            << "GPU tiled result validation successful!\n";

    } else {

        std::cout
            << "WARNING: GPU tiled checksum differs "
            << "from CPU baseline.\n";
    }

    std::cout
        << "==========================================\n";

    // --------------------------------------------------------
    // 23. Release OpenCL resources.
    // --------------------------------------------------------
    clReleaseMemObject(buffer_A);
    clReleaseMemObject(buffer_B);
    clReleaseMemObject(buffer_C);

    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return result;
}


// ============================================================
// Save benchmark results to CSV.
// ============================================================
bool save_results_to_csv(
    const std::vector<BenchmarkResult>& results
) {

    std::ofstream csv_file(
        "results/tile_sweep.csv"
    );

    if (!csv_file) {

        std::cerr
            << "Error: Could not open "
            << "results/tile_sweep.csv for writing.\n";

        return false;
    }

    csv_file
        << "implementation,"
        << "tile_size,"
        << "work_items_per_group,"
        << "matrix_size,"
        << "benchmark_runs,"
        << "average_ms,"
        << "median_ms,"
        << "min_ms,"
        << "max_ms,"
        << "checksum,"
        << "validation\n";

    csv_file
        << std::fixed
        << std::setprecision(3);

    for (const BenchmarkResult& result : results) {

        const int work_items_per_group =
            result.tile_size *
            result.tile_size;

        csv_file
            << "tiled_gpu,"
            << result.tile_size
            << ","
            << work_items_per_group
            << ","
            << MATRIX_SIZE
            << "x"
            << MATRIX_SIZE
            << ","
            << BENCHMARK_RUNS
            << ","
            << result.average_ms
            << ","
            << result.median_ms
            << ","
            << result.min_ms
            << ","
            << result.max_ms
            << ","
            << result.checksum
            << ","
            << (
                result.validation_passed
                    ? "pass"
                    : "fail"
            )
            << '\n';
    }

    csv_file.close();

    std::cout
        << "\nSweep results saved to:\n"
        << "results/tile_sweep.csv\n";

    return true;
}


// ============================================================
// Program entry point.
// ============================================================
int main(int argc, char* argv[]) {

    // --------------------------------------------------------
    // No argument:
    // Run the default 8x8 configuration.
    // --------------------------------------------------------
    if (argc <= 1) {

        BenchmarkResult result =
            run_benchmark(TILE_SIZE);

        return result.execution_success
            ? 0
            : 1;
    }

    const std::string argument =
        argv[1];

    // --------------------------------------------------------
    // Sweep mode:
    // Test 4x4, 8x8 and 16x16.
    // --------------------------------------------------------
    if (argument == "sweep") {

        const std::vector<int> tile_sizes = {
            4,
            8,
            16
        };

        std::vector<BenchmarkResult> results;

        results.reserve(
            tile_sizes.size()
        );

        for (int tile_size : tile_sizes) {

            BenchmarkResult result =
                run_benchmark(tile_size);

            if (!result.execution_success) {

                std::cerr
                    << "Error: Benchmark failed for tile size "
                    << tile_size
                    << ".\n";

                return 1;
            }

            results.push_back(result);
        }

        if (!save_results_to_csv(results)) {
            return 1;
        }

        return 0;
    }

    // --------------------------------------------------------
    // Numeric tile size:
    // Example:
    // matrix_gpu_tiled.exe 16
    // --------------------------------------------------------
    try {

        const int tile_size =
            std::stoi(argument);

        BenchmarkResult result =
            run_benchmark(tile_size);

        return result.execution_success
            ? 0
            : 1;

    } catch (const std::exception&) {

        std::cerr
            << "Invalid tile size: "
            << argument
            << '\n';

        std::cerr
            << "\nUsage:\n"
            << "  matrix_gpu_tiled.exe\n"
            << "  matrix_gpu_tiled.exe 4\n"
            << "  matrix_gpu_tiled.exe 8\n"
            << "  matrix_gpu_tiled.exe 16\n"
            << "  matrix_gpu_tiled.exe sweep\n";

        return 1;
    }
}