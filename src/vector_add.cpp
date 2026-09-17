#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

int main()
{
    // ------------------------------------------------------------
    // 1. Create input data on the host (CPU)
    // ------------------------------------------------------------

    const size_t N = 1024;

    std::vector<float> A(N);
    std::vector<float> B(N);
    std::vector<float> C(N, 0.0f);

    for (size_t i = 0; i < N; ++i)
    {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(2 * i);
    }

    // ------------------------------------------------------------
    // 2. Find OpenCL platforms
    // ------------------------------------------------------------

    cl_uint platformCount = 0;

    cl_int err = clGetPlatformIDs(
        0,
        nullptr,
        &platformCount
    );

    if (err != CL_SUCCESS || platformCount == 0)
    {
        std::cerr << "Error: No OpenCL platform found.\n";
        return 1;
    }

    std::vector<cl_platform_id> platforms(platformCount);

    err = clGetPlatformIDs(
        platformCount,
        platforms.data(),
        nullptr
    );

    if (err != CL_SUCCESS)
    {
        std::cerr << "Error: Failed to get OpenCL platforms.\n";
        return 1;
    }

    // ------------------------------------------------------------
    // 3. Select NVIDIA OpenCL platform
    // ------------------------------------------------------------

    cl_platform_id selectedPlatform = nullptr;

    for (cl_platform_id platform : platforms)
    {
        char platformName[256] = {};

        clGetPlatformInfo(
            platform,
            CL_PLATFORM_NAME,
            sizeof(platformName),
            platformName,
            nullptr
        );

        std::string name(platformName);

        std::cout << "Found platform: "
                  << name << '\n';

        if (name.find("NVIDIA") != std::string::npos ||
            name.find("CUDA") != std::string::npos)
        {
            selectedPlatform = platform;
        }
    }

    if (selectedPlatform == nullptr)
    {
        std::cerr << "Error: NVIDIA OpenCL platform not found.\n";
        return 1;
    }

    // ------------------------------------------------------------
    // 4. Find GPU device
    // ------------------------------------------------------------

    cl_uint deviceCount = 0;

    err = clGetDeviceIDs(
        selectedPlatform,
        CL_DEVICE_TYPE_GPU,
        0,
        nullptr,
        &deviceCount
    );

    if (err != CL_SUCCESS || deviceCount == 0)
    {
        std::cerr << "Error: No GPU device found.\n";
        return 1;
    }

    std::vector<cl_device_id> devices(deviceCount);

    err = clGetDeviceIDs(
        selectedPlatform,
        CL_DEVICE_TYPE_GPU,
        deviceCount,
        devices.data(),
        nullptr
    );

    if (err != CL_SUCCESS)
    {
        std::cerr << "Error: Failed to get GPU device.\n";
        return 1;
    }

    cl_device_id device = devices[0];

    char deviceName[256] = {};

    clGetDeviceInfo(
        device,
        CL_DEVICE_NAME,
        sizeof(deviceName),
        deviceName,
        nullptr
    );

    std::cout << "Using device: "
              << deviceName << '\n';

    // ------------------------------------------------------------
    // 5. Create OpenCL context
    // ------------------------------------------------------------

    cl_context context = clCreateContext(
        nullptr,
        1,
        &device,
        nullptr,
        nullptr,
        &err
    );

    if (err != CL_SUCCESS || context == nullptr)
    {
        std::cerr << "Error: Failed to create OpenCL context.\n";
        return 1;
    }

    // ------------------------------------------------------------
    // 6. Create command queue with profiling enabled
    // ------------------------------------------------------------

    cl_command_queue queue = clCreateCommandQueue(
        context,
        device,
        CL_QUEUE_PROFILING_ENABLE,
        &err
    );

    if (err != CL_SUCCESS || queue == nullptr)
    {
        std::cerr << "Error: Failed to create command queue.\n";

        clReleaseContext(context);

        return 1;
    }

    // ------------------------------------------------------------
    // 7. Read OpenCL kernel source
    // ------------------------------------------------------------

    std::ifstream kernelFile(
        "kernels/vector_add.cl"
    );

    if (!kernelFile)
    {
        std::cerr << "Error: Could not open vector_add.cl\n";

        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return 1;
    }

    std::stringstream kernelStream;

    kernelStream << kernelFile.rdbuf();

    std::string kernelSource =
        kernelStream.str();

    const char* source =
        kernelSource.c_str();

    // ------------------------------------------------------------
    // 8. Create OpenCL program
    // ------------------------------------------------------------

    cl_program program =
        clCreateProgramWithSource(
            context,
            1,
            &source,
            nullptr,
            &err
        );

    if (err != CL_SUCCESS ||
        program == nullptr)
    {
        std::cerr
            << "Error: Failed to create OpenCL program.\n";

        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return 1;
    }

    // ------------------------------------------------------------
    // 9. Build OpenCL program
    // ------------------------------------------------------------

    err = clBuildProgram(
        program,
        1,
        &device,
        nullptr,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS)
    {
        std::cerr
            << "Error: Failed to build OpenCL program.\n";

        size_t logSize = 0;

        clGetProgramBuildInfo(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            0,
            nullptr,
            &logSize
        );

        std::vector<char> buildLog(logSize);

        clGetProgramBuildInfo(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            logSize,
            buildLog.data(),
            nullptr
        );

        std::cerr << "Build log:\n";
        std::cerr << buildLog.data() << '\n';

        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return 1;
    }

    // ------------------------------------------------------------
    // 10. Create kernel object
    // ------------------------------------------------------------

    cl_kernel kernel =
        clCreateKernel(
            program,
            "vector_add",
            &err
        );

    if (err != CL_SUCCESS ||
        kernel == nullptr)
    {
        std::cerr
            << "Error: Failed to create kernel.\n";

        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return 1;
    }

    // ------------------------------------------------------------
    // 11. Create GPU buffers
    // ------------------------------------------------------------

    cl_mem bufferA =
        clCreateBuffer(
            context,
            CL_MEM_READ_ONLY,
            sizeof(float) * N,
            nullptr,
            &err
        );

    if (err != CL_SUCCESS)
    {
        std::cerr
            << "Error: Failed to create buffer A.\n";

        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return 1;
    }

    cl_mem bufferB =
        clCreateBuffer(
            context,
            CL_MEM_READ_ONLY,
            sizeof(float) * N,
            nullptr,
            &err
        );

    if (err != CL_SUCCESS)
    {
        std::cerr
            << "Error: Failed to create buffer B.\n";

        clReleaseMemObject(bufferA);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return 1;
    }

    cl_mem bufferC =
        clCreateBuffer(
            context,
            CL_MEM_WRITE_ONLY,
            sizeof(float) * N,
            nullptr,
            &err
        );

    if (err != CL_SUCCESS)
    {
        std::cerr
            << "Error: Failed to create buffer C.\n";

        clReleaseMemObject(bufferA);
        clReleaseMemObject(bufferB);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return 1;
    }

    // ------------------------------------------------------------
    // 12. Copy input data from CPU to GPU
    // ------------------------------------------------------------

    err = clEnqueueWriteBuffer(
        queue,
        bufferA,
        CL_TRUE,
        0,
        sizeof(float) * N,
        A.data(),
        0,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS)
    {
        std::cerr
            << "Error: Failed to copy A to GPU.\n";

        return 1;
    }

    err = clEnqueueWriteBuffer(
        queue,
        bufferB,
        CL_TRUE,
        0,
        sizeof(float) * N,
        B.data(),
        0,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS)
    {
        std::cerr
            << "Error: Failed to copy B to GPU.\n";

        return 1;
    }

    // ------------------------------------------------------------
    // 13. Set kernel arguments
    // ------------------------------------------------------------

    err = clSetKernelArg(
        kernel,
        0,
        sizeof(cl_mem),
        &bufferA
    );

    if (err != CL_SUCCESS)
    {
        std::cerr
            << "Error: Failed to set kernel argument A.\n";

        return 1;
    }

    err = clSetKernelArg(
        kernel,
        1,
        sizeof(cl_mem),
        &bufferB
    );

    if (err != CL_SUCCESS)
    {
        std::cerr
            << "Error: Failed to set kernel argument B.\n";

        return 1;
    }

    err = clSetKernelArg(
        kernel,
        2,
        sizeof(cl_mem),
        &bufferC
    );

    if (err != CL_SUCCESS)
    {
        std::cerr
            << "Error: Failed to set kernel argument C.\n";

        return 1;
    }

    // ------------------------------------------------------------
    // 14. Define global work size
    // ------------------------------------------------------------

    size_t globalWorkSize = N;

    // ------------------------------------------------------------
    // 15. Launch kernel and collect profiling event
    // ------------------------------------------------------------

    cl_event kernelEvent = nullptr;

    err = clEnqueueNDRangeKernel(
        queue,
        kernel,
        1,
        nullptr,
        &globalWorkSize,
        nullptr,
        0,
        nullptr,
        &kernelEvent
    );

    if (err != CL_SUCCESS)
    {
        std::cerr
            << "Error: Failed to launch kernel.\n";

        return 1;
    }

    // Wait for GPU execution to finish.
    clFinish(queue);

    // ------------------------------------------------------------
    // 16. Get GPU kernel execution time
    // ------------------------------------------------------------

    cl_ulong startTime = 0;
    cl_ulong endTime = 0;

    err = clGetEventProfilingInfo(
        kernelEvent,
        CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong),
        &startTime,
        nullptr
    );

    if (err != CL_SUCCESS)
    {
        std::cerr
            << "Error: Failed to get kernel start time.\n";

        return 1;
    }

    err = clGetEventProfilingInfo(
        kernelEvent,
        CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong),
        &endTime,
        nullptr
    );

    if (err != CL_SUCCESS)
    {
        std::cerr
            << "Error: Failed to get kernel end time.\n";

        return 1;
    }

    double kernelTimeMs =
        static_cast<double>(
            endTime - startTime
        ) / 1'000'000.0;

    std::cout
        << "GPU kernel time: "
        << kernelTimeMs
        << " ms\n";

    // ------------------------------------------------------------
    // 17. Copy result from GPU to CPU
    // ------------------------------------------------------------

    err = clEnqueueReadBuffer(
        queue,
        bufferC,
        CL_TRUE,
        0,
        sizeof(float) * N,
        C.data(),
        0,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS)
    {
        std::cerr
            << "Error: Failed to copy result from GPU.\n";

        return 1;
    }

    // ------------------------------------------------------------
    // 18. Verify result
    // ------------------------------------------------------------

    bool correct = true;

    for (size_t i = 0; i < N; ++i)
    {
        float expected = A[i] + B[i];

        if (C[i] != expected)
        {
            correct = false;

            std::cerr
                << "Mismatch at index "
                << i
                << ": expected "
                << expected
                << ", got "
                << C[i]
                << '\n';

            break;
        }
    }

    if (correct)
    {
        std::cout
            << "Vector addition successful!\n";

        std::cout
            << "All "
            << N
            << " results are correct.\n";
    }

    // ------------------------------------------------------------
    // 19. Release OpenCL resources
    // ------------------------------------------------------------

    clReleaseEvent(kernelEvent);

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);

    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return correct ? 0 : 1;
}
