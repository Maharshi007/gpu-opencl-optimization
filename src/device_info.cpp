#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::string load_kernel_source(const std::string& path)
{
    std::ifstream file(path);

    if (!file)
    {
        throw std::runtime_error("Failed to open kernel file: " + path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    return buffer.str();
}

void print_kernel_info(
    cl_context context,
    cl_device_id device,
    const std::string& kernel_path,
    const std::string& kernel_name,
    const char* build_options
)
{
    std::cout << "\nKernel: " << kernel_name << '\n';
    std::cout << "Source: " << kernel_path << '\n';

    std::string source = load_kernel_source(kernel_path);

    const char* source_ptr = source.c_str();
    size_t source_size = source.size();

    cl_int err = CL_SUCCESS;

    cl_program program = clCreateProgramWithSource(
        context,
        1,
        &source_ptr,
        &source_size,
        &err
    );

    if (err != CL_SUCCESS)
    {
        std::cerr << "Failed to create program.\n";
        return;
    }

    err = clBuildProgram(
        program,
        1,
        &device,
        build_options,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS)
    {
        std::cerr << "Failed to build kernel.\n";

        size_t log_size = 0;

        clGetProgramBuildInfo(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            0,
            nullptr,
            &log_size
        );

        std::string log(log_size, '\0');

        clGetProgramBuildInfo(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            log_size,
            log.data(),
            nullptr
        );

        std::cerr << log << '\n';

        clReleaseProgram(program);
        return;
    }

    cl_kernel kernel = clCreateKernel(
        program,
        kernel_name.c_str(),
        &err
    );

    if (err != CL_SUCCESS)
    {
        std::cerr << "Failed to create kernel.\n";
        clReleaseProgram(program);
        return;
    }

    size_t max_work_group_size = 0;

    clGetKernelWorkGroupInfo(
        kernel,
        device,
        CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(max_work_group_size),
        &max_work_group_size,
        nullptr
    );

    size_t preferred_multiple = 0;

    clGetKernelWorkGroupInfo(
        kernel,
        device,
        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        sizeof(preferred_multiple),
        &preferred_multiple,
        nullptr
    );

    cl_ulong local_memory = 0;

    clGetKernelWorkGroupInfo(
        kernel,
        device,
        CL_KERNEL_LOCAL_MEM_SIZE,
        sizeof(local_memory),
        &local_memory,
        nullptr
    );

    std::cout << "Max work-group size: "
              << max_work_group_size << '\n';

    std::cout << "Preferred work-group size multiple: "
              << preferred_multiple << '\n';

    std::cout << "Kernel local memory used: "
              << local_memory / 1024
              << " bytes\n";

    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

int main()
{
    cl_uint num_platforms = 0;

    cl_int err = clGetPlatformIDs(
        0,
        nullptr,
        &num_platforms
    );

    if (err != CL_SUCCESS || num_platforms == 0)
    {
        std::cerr << "No OpenCL platforms found.\n";
        return 1;
    }

    std::vector<cl_platform_id> platforms(num_platforms);

    clGetPlatformIDs(
        num_platforms,
        platforms.data(),
        nullptr
    );

    for (cl_platform_id platform : platforms)
    {
        size_t platform_name_size = 0;

        clGetPlatformInfo(
            platform,
            CL_PLATFORM_NAME,
            0,
            nullptr,
            &platform_name_size
        );

        std::string platform_name(platform_name_size, '\0');

        clGetPlatformInfo(
            platform,
            CL_PLATFORM_NAME,
            platform_name_size,
            platform_name.data(),
            nullptr
        );

        cl_uint num_devices = 0;

        err = clGetDeviceIDs(
            platform,
            CL_DEVICE_TYPE_GPU,
            0,
            nullptr,
            &num_devices
        );

        if (err != CL_SUCCESS || num_devices == 0)
        {
            continue;
        }

        std::vector<cl_device_id> devices(num_devices);

        clGetDeviceIDs(
            platform,
            CL_DEVICE_TYPE_GPU,
            num_devices,
            devices.data(),
            nullptr
        );

        for (cl_device_id device : devices)
        {
            size_t device_name_size = 0;

            clGetDeviceInfo(
                device,
                CL_DEVICE_NAME,
                0,
                nullptr,
                &device_name_size
            );

            std::string device_name(device_name_size, '\0');

            clGetDeviceInfo(
                device,
                CL_DEVICE_NAME,
                device_name_size,
                device_name.data(),
                nullptr
            );

            std::cout << "\n========================================\n";
            std::cout << "Platform: " << platform_name << '\n';
            std::cout << "Device: " << device_name << '\n';
            std::cout << "========================================\n";
            cl_uint compute_units = 0;
            cl_ulong local_mem_size = 0;
            cl_uint max_work_item_dimensions = 0;
            cl_ulong global_mem_size = 0;

            clGetDeviceInfo(
                device,
                CL_DEVICE_MAX_COMPUTE_UNITS,
                sizeof(compute_units),
                &compute_units,
                nullptr
            );

            clGetDeviceInfo(
                device,
                CL_DEVICE_LOCAL_MEM_SIZE,
                sizeof(local_mem_size),
                &local_mem_size,
                nullptr
            );

            clGetDeviceInfo(
                device,
                CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                sizeof(max_work_item_dimensions),
                &max_work_item_dimensions,
                nullptr
            );

            clGetDeviceInfo(
                device,
                CL_DEVICE_GLOBAL_MEM_SIZE,
                sizeof(global_mem_size),
                &global_mem_size,
                nullptr
            );

            std::cout << "Compute units: "
                      << compute_units << '\n';

            std::cout << "Device local memory: "
                      << local_mem_size / 1024
                      << " KB\n";

            std::cout << "Max work-item dimensions: "
                      << max_work_item_dimensions << '\n';

            std::cout << "Global memory: "
                      << global_mem_size / (1024 * 1024)
                      << " MB\n";
            cl_context context = clCreateContext(
                nullptr,
                1,
                &device,
                nullptr,
                nullptr,
                &err
            );

            if (err != CL_SUCCESS)
            {
                std::cerr << "Failed to create OpenCL context.\n";
                continue;
            }

            // Inspect the naive matrix multiplication kernel.
            print_kernel_info(
                context,
                device,
                "kernels/matrix_mul.cl",
                "matrix_multiply",
                nullptr
            );

            // Inspect the tiled matrix multiplication kernel.
            print_kernel_info(
                context,
                device,
                "kernels/matrix_mul_tiled.cl",
                "matrix_multiply_tiled",
                "-DTILE_SIZE=8"
            );

            clReleaseContext(context);
        }
    }

    return 0;
}