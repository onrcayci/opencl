#ifndef OPENCL_H
#define OPENCL_H

#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>

typedef struct clSetup
{
    cl_platform_id *platforms;
    cl_device_id *devices;
    cl_context *ctx;
} clSetup;

clSetup *createClSetup();

void releaseClSetup(clSetup *setup);

void check_opencl_error(
    cl_int err,
    char *err_msg,
    void (*additional_err_fn)(cl_program program, cl_device_id device),
    cl_program program,
    cl_device_id device);

void setup_opencl(
    cl_device_type device_type,
    clSetup *setup);

char *read_opencl_kernel_file(char *file_path);

#endif