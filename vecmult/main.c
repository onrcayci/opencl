#define CL_VECMULT_FILE "vecmult.cl"
#define CL_VECMULT_KERNEL "vecmult"

#include "../lib/opencl.h"

#include <stdio.h>
#include <stdlib.h>

void get_program_build_log(cl_program program, cl_device_id device);

int main()
{
    // initialize the OpenCL setup object
    clSetup *setup = createClSetup();

    // initialize the OpenCL error object
    cl_int err = CL_SUCCESS;

    // create an OpenCL setup
    setup_opencl(CL_DEVICE_TYPE_GPU, setup);

    // read the kernel file
    char *buffer = read_opencl_kernel_file(CL_VECMULT_FILE);

    // create program
    cl_program program = clCreateProgramWithSource(*(setup->ctx), 1, (const char **)&buffer, NULL, &err);
    check_opencl_error(err, "Error creating the program", NULL, NULL, NULL);
    free(buffer);

    // compile and link the program
    err = clBuildProgram(program, 1, setup->devices, NULL, NULL, NULL);
    check_opencl_error(err, "Error building the program", &get_program_build_log, program, *(setup->devices));

    // create the kernel
    cl_kernel kernel = clCreateKernel(program, CL_VECMULT_KERNEL, &err);
    check_opencl_error(err, "Error creating the kernel", NULL, NULL, NULL);

    // create the queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(*(setup->ctx), *(setup->devices), NULL, &err);
    check_opencl_error(err, "Error creating the queue", NULL, NULL, NULL);

    // initialize the data for compute
    float *vec1 = calloc(4, sizeof(float));
    float *vec2 = calloc(4, sizeof(float));
    float *result = calloc(4, sizeof(float));

    // populate the vectors
    for (int i = 0; i < 4; i++)
    {
        vec1[i] = i;
        vec2[i] = 3 * i;
    }

    // send the vectors to the device memory
    cl_mem dev_vec1 = clCreateBuffer(*(setup->ctx), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 4, vec1, &err);
    check_opencl_error(err, "Error creating the device buffer", NULL, NULL, NULL);
    cl_mem dev_vec2 = clCreateBuffer(*(setup->ctx), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 4, vec2, &err);
    check_opencl_error(err, "Error creating the device buffer", NULL, NULL, NULL);
    cl_mem dev_vec3 = clCreateBuffer(*(setup->ctx), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 4, result, &err);
    check_opencl_error(err, "Error creating the device buffer", NULL, NULL, NULL);

    // add the variables in device memory to the kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dev_vec1);
    check_opencl_error(err, "Error setting the kernel argument", NULL, NULL, NULL);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dev_vec2);
    check_opencl_error(err, "Error setting the kernel argument", NULL, NULL, NULL);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dev_vec3);
    check_opencl_error(err, "Error setting the kernel argument", NULL, NULL, NULL);

    // define the number of workers that we want to use
    size_t work_units = 4;

    // run the function on the device
    err = clEnqueueNDRangeKernel(queue, kernel, 1, 0, &work_units, 0, 0, NULL, NULL);
    check_opencl_error(err, "Error enqueuing the kernel", NULL, NULL, NULL);

    // get the result back from the device to the host
    err = clEnqueueReadBuffer(queue, dev_vec3, CL_TRUE, 0, sizeof(float) * 4, result, 0, NULL, NULL);
    check_opencl_error(err, "Error reading the device buffer", NULL, NULL, NULL);

    // print the result
    for (int i = 0; i < 4; i++)
    {
        printf("%.2f\n", result[i]);
    }

    // Release the memory that is held by the OpenCL variables
    err = clReleaseMemObject(dev_vec1);
    check_opencl_error(err, "Error releasing the memory", NULL, NULL, NULL);
    err = clReleaseMemObject(dev_vec2);
    check_opencl_error(err, "Error releasing the memory", NULL, NULL, NULL);
    err = clReleaseMemObject(dev_vec3);
    check_opencl_error(err, "Error releasing the memory", NULL, NULL, NULL);
    err = clReleaseKernel(kernel);
    check_opencl_error(err, "Error releasing the kernel", NULL, NULL, NULL);
    err = clReleaseCommandQueue(queue);
    check_opencl_error(err, "Error releasing the queue", NULL, NULL, NULL);
    err = clReleaseProgram(program);
    check_opencl_error(err, "Error releasing the program", NULL, NULL, NULL);
    err = clReleaseContext(*(setup->ctx));
    check_opencl_error(err, "Error releasing the context", NULL, NULL, NULL);

    releaseClSetup(setup);

    return 0;
}

void get_program_build_log(cl_program program, cl_device_id device)
{
    size_t log_size = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log_msg = calloc(log_size, sizeof(char));
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log_msg, NULL);
    printf("%s\n", log_msg);
    free(log_msg);
}