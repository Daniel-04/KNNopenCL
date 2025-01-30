#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifndef DEVICE_TYPE
#define DEVICE_TYPE CL_DEVICE_TYPE_GPU
#endif

#define RAW(...) (#__VA_ARGS__)
#define ENTRYNAME "kern"

typedef struct pipeline {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
} pipeline;

void setup_pipeline(pipeline *pipe) {
  cl_int err;

  cl_uint num_platforms;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Platform error? %d\n", err);
    abort();
  }
  cl_platform_id *platforms = malloc(sizeof(cl_platform_id) * num_platforms);
  err = clGetPlatformIDs(num_platforms, platforms, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Platform error? %d\n", err);
    abort();
  }
  err = CL_FALSE;
  for (int i = 0; i < num_platforms; i++) {
    cl_uint num_devices;
    clGetDeviceIDs(platforms[i], DEVICE_TYPE, 0, NULL, &num_devices);
    if (num_devices > 0) {
      pipe->platform = platforms[i];
      err = CL_SUCCESS;
      break;
    }
  }
  free(platforms);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "No platform found with device: %d\n", DEVICE_TYPE);
    abort();
  }

  err = clGetDeviceIDs(pipe->platform, DEVICE_TYPE, 1, &pipe->device, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Device error? %d\n", err);
    abort();
  }
  pipe->context = clCreateContext(NULL, 1, &pipe->device, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Context creation error? %d\n", err);
    abort();
  }
  pipe->queue = clCreateCommandQueueWithProperties(pipe->context, pipe->device,
                                                   NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Command queue creation error? %d\n", err);
    abort();
  }
}

cl_mem create_buffer(pipeline *pipe, cl_mem_flags flags, size_t size,
                     void *host_ptr) {
  cl_int err;
  cl_mem out = clCreateBuffer(pipe->context, flags, size, host_ptr, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Buffer creation error? %d\n", err);
    abort();
  }

  return out;
}

cl_kernel compile_kernel(pipeline *pipe, const char *kernel_source) {
  cl_int err;
  cl_program program =
      clCreateProgramWithSource(pipe->context, 1, &kernel_source, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Program creation error? %d\n", err);
    abort();
  }
  clBuildProgram(program, 1, &pipe->device, NULL, NULL, NULL);

  cl_kernel kernel = clCreateKernel(program, ENTRYNAME, &err);
  if (err == CL_INVALID_KERNEL_NAME) {
    fprintf(stderr,
            "Kernel creation error %d, expected entry point name '%s'\n", err,
            ENTRYNAME);
    abort();
  } else if (err != CL_SUCCESS) {
    fprintf(stderr, "Kernel creation error? %d\n", err);
    abort();
  }
  return kernel;
}

void release_pipeline(pipeline *pipe) {
  clReleaseDevice(pipe->device);
  clReleaseContext(pipe->context);
  clReleaseCommandQueue(pipe->queue);
}
