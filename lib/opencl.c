#include <stdio.h>

#include "opencl.h"

clSetup *createClSetup()
{
    clSetup *setup = malloc(sizeof(clSetup));
    setup->ctx = calloc(1, sizeof(cl_context));
    setup->devices = calloc(1, sizeof(cl_device_id));
    setup->platforms = calloc(1, sizeof(cl_platform_id));
    return setup;
}

void releaseClSetup(clSetup *setup)
{
    free(setup->ctx);
    free(setup->devices);
    free(setup->platforms);
    free(setup);
}

void check_opencl_error(
    cl_int err,
    char *err_msg,
    void (*additional_err_fn)(cl_program program, cl_device_id device),
    cl_program program,
    cl_device_id device)
{
    if (err != CL_SUCCESS)
    {
        printf("%s: %d\n", err_msg, err);
        if (additional_err_fn != NULL)
        {
            (*additional_err_fn)(program, device);
        }
        exit(1);
    }
}

void setup_opencl(
    cl_device_type device_type,
    clSetup *setup)
{
    cl_int err = CL_SUCCESS;
    err = clGetPlatformIDs(1, setup->platforms, NULL);
    check_opencl_error(err, "Error getting the platform id", NULL, NULL, NULL);
    err = clGetDeviceIDs(
        *(setup->platforms),
        device_type,
        1,
        setup->devices, NULL);
    char *err_msg = calloc(64, sizeof(char));
    sprintf(
        err_msg,
        "Error getting the device id for platform %d",
        *(setup->platforms));
    check_opencl_error(
        err,
        err_msg,
        NULL,
        NULL,
        NULL);

    *(setup->ctx) = clCreateContext(
        NULL,
        1,
        setup->devices,
        NULL,
        NULL,
        &err);
    sprintf(
        err_msg,
        "Error creating context for platform %d and device %d\n",
        *(setup->platforms),
        *(setup->devices));
    check_opencl_error(
        err,
        err_msg,
        NULL,
        NULL,
        NULL);
    free(err_msg);
}

char *read_opencl_kernel_file(char *file_path)
{
    FILE *kernel_file = fopen(file_path, "r");
    fseek(kernel_file, 0, SEEK_END);
    size_t filesize = ftell(kernel_file);
    rewind(kernel_file);
    char *buffer = calloc(filesize + 1, sizeof(char));
    fread(buffer, sizeof(char), filesize, kernel_file);
    buffer[filesize] = '\0';
    fclose(kernel_file);
    return buffer;
}