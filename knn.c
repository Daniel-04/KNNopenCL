#include <string.h>
#include <time.h>
#define CL_TARGET_OPENCL_VERSION 300
#include "cl-helpers.c"
#include "csv-helpers.c"
#include <CL/cl.h>
#include <stdlib.h>

const char *knn_src = RAW(__kernel void kern(
    __global const float *train_points, __global float *test_points,
    const int train_npoints, const int test_npoints, const int num_features,
    const int k) {
  int global_id = get_global_id(0);
  if (global_id >= test_npoints)
    return;
  const int MAX_K = 16;
  float nearest_dists[MAX_K];
  float nearest_classes[MAX_K];
  for (int i = 0; i < k; i++) {
    nearest_dists[i] = INFINITY;
  }

  int train_features = num_features - 1;
  for (int i = 0; i < train_npoints; i++) {
    float dist = 0;
    for (int j = 0; j < train_features; j++) {
      float diff = test_points[global_id * num_features + j] -
                   train_points[i * num_features + j];
      dist += diff * diff;
    }
    float class = train_points[i * num_features + train_features];

    int inserted = 0;
    int position = 1;
    while (dist < nearest_dists[k - 1] && inserted == 0) {
      if (k - position == 0) {
        nearest_dists[k - position] = dist;
        nearest_classes[k - position] = class;
        inserted = 1;
      } else if (dist < nearest_dists[k - position - 1]) {
        nearest_dists[k - position] = nearest_dists[k - position - 1];
        nearest_classes[k - position] = nearest_classes[k - position - 1];
        position++;
      } else {
        nearest_dists[k - position] = dist;
        nearest_classes[k - position] = class;
        inserted = 1;
      }
    }
  }

  float majority = nearest_classes[k - 1];
  int count = 1;
  for (int i = k - 2; i > 0; i--) {
    if (nearest_classes[i] == majority) {
      count++;
    } else {
      count--;
      if (count == 0) {
        majority = nearest_classes[i];
        count = 1;
      }
    }
  }

  test_points[global_id * num_features + train_features] = majority;
});

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "%s train_file test_file [out_file] [k]\n", argv[0]);
    abort();
  }
  const char *trainfile = argv[1];
  const char *testfile = argv[2];
  const char *outfile = argc >= 4 ? argv[3] : NULL;
  const int k = argc >= 5 ? atoi(argv[4]) : 3;
  pipeline pipe;
  setup_pipeline(&pipe);

  size_t train_npoints = readPoints(trainfile);
  size_t train_nfeatures = readFeatures(trainfile);
  float *train_data = readData(trainfile, train_npoints, train_nfeatures);
  size_t test_npoints = readPoints(testfile);
  size_t test_nfeatures = readFeatures(testfile);
  float *test_data = readData(testfile, test_npoints, test_nfeatures);

  // device memory
  cl_mem d_train_data = create_buffer(
      &pipe, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(float) * train_npoints * train_nfeatures, train_data);
  cl_mem d_test_data =
      create_buffer(&pipe, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                    sizeof(float) * test_npoints * test_nfeatures, test_data);

  // kernel
  cl_kernel knn_kernel = compile_kernel(&pipe, knn_src);
  clSetKernelArg(knn_kernel, 0, sizeof(cl_mem), &d_train_data);
  clSetKernelArg(knn_kernel, 1, sizeof(cl_mem), &d_test_data);
  clSetKernelArg(knn_kernel, 2, sizeof(int), &train_npoints);
  clSetKernelArg(knn_kernel, 3, sizeof(int), &test_npoints);
  clSetKernelArg(knn_kernel, 4, sizeof(int), &train_nfeatures);
  clSetKernelArg(knn_kernel, 5, sizeof(int), &k);

  // execution
  clFinish(pipe.queue);
  double start = clock();
  cl_int err = clEnqueueNDRangeKernel(pipe.queue, knn_kernel, 1, NULL,
                                      &test_npoints, NULL, 0, NULL, NULL);
  clFinish(pipe.queue);
  double end = clock();
  printf("Kernel took: %lf seconds\n", (end - start) / CLOCKS_PER_SEC);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Kernel enqueue error? %d\n", err);
    abort();
  }

  clEnqueueReadBuffer(pipe.queue, d_test_data, CL_TRUE, 0,
                      sizeof(float) * test_npoints * test_nfeatures, test_data,
                      0, NULL, NULL);
  if (outfile) {
    printf("Output written to %s\n", outfile);
    writeData(outfile, test_npoints, test_nfeatures, test_data);
  }

  clReleaseMemObject(d_train_data);
  clReleaseMemObject(d_test_data);
  release_pipeline(&pipe);
  free(train_data);
  free(test_data);
  return 0;
}
