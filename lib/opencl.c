#include <stdio.h>
#include <stdint.h>

#include "opencl.h"

/**
 * clSetup constructor.
 *
 * returns: a pointer to an empty clSetup struct.
 */
clSetup *createClSetup()
{
    clSetup *setup = malloc(sizeof(clSetup));
    setup->ctx = calloc(1, sizeof(cl_context));
    setup->devices = calloc(1, sizeof(cl_device_id));
    setup->platforms = calloc(1, sizeof(cl_platform_id));
    return setup;
}

/**
 * clSetup desctuctor.
 */
void releaseClSetup(clSetup *setup)
{
    free(setup->ctx);
    free(setup->devices);
    free(setup->platforms);
    free(setup);
}

/**
 * Check whether an OpenCL procedure returned an error.
 *
 * cl_int err: the error object that is returned by an OpenCL procedure.
 * char *err_msg: the error message that will be printed.
 * void (*additional_err_fn)(cl_program program, cl_device_id device): function
 * pointer that is used to do additional things if an error is found.
 * cl_program program: the OpenCL program.
 * cl_device_id device: the OpenCL compatible device.
 */
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

/**
 * Setup an OpenCL context for the given device type.
 *
 * cl_device_type device_type: the type of the OpenCL compatible device.
 * clSetup *setup: the setup object that has the OpenCL context, platform id and device id.
 */
void setup_opencl(
    cl_device_type device_type,
    clSetup *setup)
{
    // initialize the error object
    cl_int err = CL_SUCCESS;

    // initialize the error message string
    char *err_msg = calloc(64, sizeof(char));

    // check if there are any OpenCL compatible platforms
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(INT8_MAX, NULL, &num_platforms);
    check_opencl_error(err, "Error getting the number of platforms", NULL, NULL, NULL);

    // if the number of platforms is 0, exit the program
    if (num_platforms == 0)
    {
        printf("No OpenCL platforms are detected.");
        exit(1);
    }

    // get the platform id
    err = clGetPlatformIDs(1, setup->platforms, NULL);
    check_opencl_error(err, "Error getting the platform id", NULL, NULL, NULL);

    // get the device id
    err = clGetDeviceIDs(
        *(setup->platforms),
        device_type,
        1,
        setup->devices, NULL);
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

    // create OpenCL context for the device
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

    // free up the error message string since we are done with the setup
    free(err_msg);
}

/**
 * Read an OpenCL kernel file.
 *
 * char *file_path: The file path of the kernel file that will be read.
 * returns: the buffer that holds the contents of the kernel file.
 */
char *read_opencl_kernel_file(char *file_path)
{
    FILE *kernel_file = fopen(file_path, "r");

    // get the file size to initialize a buffer of that size
    fseek(kernel_file, 0, SEEK_END);
    size_t filesize = ftell(kernel_file);
    rewind(kernel_file);

    // allocate memory for the buffer
    char *buffer = calloc(filesize + 1, sizeof(char));

    // load the contents of the kernel file to the memory
    fread(buffer, sizeof(char), filesize, kernel_file);

    // set the end of the file to NULL for OpenCL
    buffer[filesize] = '\0';

    fclose(kernel_file);
    return buffer;
}