#include <stdio.h>
#include <stdlib.h>

size_t readPoints(const char *filename) {
  FILE *infile = fopen(filename, "r");
  if (infile == NULL) {
    fprintf(stderr, "File opening error\n");
    abort();
  }

  size_t linecount = 0;
  for (int c = fgetc(infile); c != EOF; c = fgetc(infile)) {
    if (c == '\n')
      linecount++;
  }

  fclose(infile);
  return linecount;
}

size_t readFeatures(const char *filename) {
  FILE *infile = fopen(filename, "r");
  if (infile == NULL) {
    fprintf(stderr, "File opening error\n");
    abort();
  }

  size_t fieldcount = 1;
  for (int c = fgetc(infile); c != EOF && c != '\n'; c = fgetc(infile)) {
    if (c == ',')
      fieldcount++;
  }

  fclose(infile);
  return fieldcount;
}

float *readData(const char *filename, size_t npoints, size_t nfeatures) {
  size_t nfields = npoints * nfeatures;
  float *data = malloc(nfields * sizeof(float));
  FILE *infile = fopen(filename, "r");
  if (infile == NULL) {
    fprintf(stderr, "File opening error\n");
    abort();
  }

  for (size_t i = 0; i < nfields; i++) {
    fscanf(infile, "%f,", &data[i]);
  }

  fclose(infile);
  return data;
}

void writeData(const char *filename, size_t npoints, size_t nfeatures,
               float *data) {
  FILE *outfile = fopen(filename, "w");
  if (outfile == NULL) {
    fprintf(stderr, "File opening error\n");
    abort();
  }

  for (size_t i = 0; i < npoints; i++) {
    for (size_t j = 0; j < nfeatures; j++) {
      fprintf(outfile, "%f,", data[i * nfeatures + j]);
    }
    fprintf(outfile, "\n");
  }

  fclose(outfile);
}
