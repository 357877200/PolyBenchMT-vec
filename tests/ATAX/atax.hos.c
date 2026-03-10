#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include "hthread_host.h"
#include "../common/tool.h"

#define PERCENT_DIFF_ERROR_THRESHOLD 0.5
#ifndef M_PI
#define M_PI 3.14159
#endif

// CPU implementation for ATax kernels
void atax_cpu1(int nx, int ny, double *A, double *x, double *tmp) {
    int i, j;
    for (i = 0; i < nx; i++) {
        tmp[i] = 0;
        for (j = 0; j < ny; j++) {
            tmp[i] += A[i * ny + j] * x[j];
        }
    }
}

void atax_cpu2(int nx, int ny, double *A, double *y, double *tmp) {
    int i, j;
    for (j = 0; j < ny; j++) {
        y[j] = 0;
    }
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            y[j] += A[i * ny + j] * tmp[i];
        }
    }
}

// Initialize input arrays
void init_array(int nx, int ny, double *x, double *A) {
    int i, j;
    for (i = 0; i < nx; i++) {
        x[i] = i * M_PI;
        for (j = 0; j < ny; j++) {
            A[i * ny + j] = ((double)i * j) / nx;
        }
    }
}

// Check results between CPU and DSP
int check_result(int ny, double *y_host, double *y_device) {
    int errNum = 0;
    for (int i = 0; i < ny; i++) {
        if (percentDiff(y_host[i], y_device[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
            if (errNum < 10) {
                fprintf(stderr, "y data error at (%d): Host=%.2f, Device=%.2f\n", i, y_host[i], y_device[i]);
            }
            errNum++;
        }
    }
    if (errNum != 0) {
        fprintf(stderr, "Total errors: %d\n", errNum);
    }
    return errNum;
}

// Save performance data
void save_data(int nx, int ny, uint64_t *before_hot_data, uint64_t *after_hot_data, uint64_t timeDsp, uint64_t timeCpu, int clusterId, char *devProgram, int nthreads, char *kernel) {
    FILE *file = fopen("tests/ATAX/atax_events.txt", "a");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }
    fprintf(file, "%d,%d,%s,%s,%d,Difference,", clusterId, nx * ny, devProgram, kernel, nthreads);
    for (int eid = 0; eid < 26; eid++) {
        fprintf(file, "%lu", after_hot_data[eid] - before_hot_data[eid]);
        if (eid < 25) fprintf(file, ",");
    }
    fprintf(file, ",%fs,%fs\n", timeDsp / 1e6, timeCpu / 1e6);
    fclose(file);
}

int main(int argc, char **argv) {
    int retc;
    int clusterId = 1;
    int nx = 4096; // Default problem size for x dimension
    int ny = 4096; // Default problem size for y dimension
    char *devProgram = "operators/ATAX/atax.dev.dat";
    int nthreads = 1;
    char *kernel1 = "atax_kernel1";
    char *kernel2 = "atax_kernel2";
    uint64_t timeDsp1 = 0, timeDsp2 = 0;
    uint64_t timeCpu1 = 0, timeCpu2 = 0;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (i + 1 < argc) {
            if (strcmp(argv[i], "--clusterId") == 0 || strcmp(argv[i], "-c") == 0) {
                clusterId = atoi(argv[i + 1]);
                i++;
            } else if (strcmp(argv[i], "--nx") == 0) {
                nx = atoi(argv[i + 1]);
                i++;
            } else if (strcmp(argv[i], "--ny") == 0) {
                ny = atoi(argv[i + 1]);
                i++;
            } else if (strcmp(argv[i], "--program") == 0 || strcmp(argv[i], "-p") == 0) {
                devProgram = argv[i + 1];
                i++;
            } else if (strcmp(argv[i], "--threads") == 0 || strcmp(argv[i], "-t") == 0) {
                nthreads = atoi(argv[i + 1]);
                i++;
            } else if (strcmp(argv[i], "--kernel1") == 0 || strcmp(argv[i], "-k1") == 0) {
                kernel1 = argv[i + 1];
                i++;
            } else if (strcmp(argv[i], "--kernel2") == 0 || strcmp(argv[i], "-k2") == 0) {
                kernel2 = argv[i + 1];
                i++;
            }
        }
    }

    // Device initialization and parameter validation
    if (clusterId < 0 || clusterId > 3) { fprintf(stderr, "invalid clusterId: %d\n", clusterId); return 2; }
    if (nthreads <= 0) { fprintf(stderr, "invalid nthreads: %d\n", nthreads); return 2; }
    if (access(devProgram, F_OK) != 0) { fprintf(stderr, "%s: No such file or directory\n", devProgram); return 2; }

    retc = hthread_dev_open(clusterId); if (retc != 0) { fprintf(stderr, "Failed to open device\n"); return retc; }
    retc = hthread_dat_load(clusterId, devProgram); if (retc != 0) { fprintf(stderr, "Failed to load program\n"); return retc; }
    int availThreads = hthread_get_avail_threads(clusterId);
    if (nthreads > availThreads) {
        fprintf(stderr, "number of threads overflow: avail %d, actual %d\n", availThreads, nthreads);
        hthread_dat_unload(clusterId);
        hthread_dev_close(clusterId);
        return 2;
    }

    // Memory allocation
    size_t sizeA = (size_t)nx * ny * sizeof(double);
    size_t sizex = (size_t)nx * sizeof(double);
    size_t sizey = (size_t)ny * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    // Device memory allocation
    double *A = (double *)hthread_malloc(clusterId, sizeA, HT_MEM_RO); if (!A) { fprintf(stderr, "Malloc A failed\n"); return 1; }
    double *x = (double *)hthread_malloc(clusterId, sizex, HT_MEM_RO); if (!x) { fprintf(stderr, "Malloc x failed\n"); return 1; }
    double *y_dsp = (double *)hthread_malloc(clusterId, sizey, HT_MEM_RW); if (!y_dsp) { fprintf(stderr, "Malloc y_dsp failed\n"); return 1; }
    double *tmp = (double *)hthread_malloc(clusterId, sizex, HT_MEM_RW); if (!tmp) { fprintf(stderr, "Malloc tmp failed\n"); return 1; }

    // Host memory allocation (for validation)
    double *A_host = (double *)malloc(sizeA); if (!A_host) { fprintf(stderr, "Malloc A_host failed\n"); return 1; }
    double *x_host = (double *)malloc(sizex); if (!x_host) { fprintf(stderr, "Malloc x_host failed\n"); return 1; }
    double *y_host = (double *)malloc(sizey); if (!y_host) { fprintf(stderr, "Malloc y_host failed\n"); return 1; }
    double *tmp_host = (double *)malloc(sizex); if (!tmp_host) { fprintf(stderr, "Malloc tmp_host failed\n"); return 1; }

    // Event counter memory allocation for each kernel
    uint64_t *before_hot_data1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!before_hot_data1) { return 1; }
    uint64_t *after_hot_data1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!after_hot_data1) { return 1; }
    uint64_t *before_hot_data2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!before_hot_data2) { return 1; }
    uint64_t *after_hot_data2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!after_hot_data2) { return 1; }

    // Data initialization
    init_array(nx, ny, x, A);
    memcpy(A_host, A, sizeA);
    memcpy(x_host, x, sizex);
    memset(y_dsp, 0, sizey);
    memset(tmp, 0, sizex);
    memset(y_host, 0, sizey);
    memcpy(tmp_host, tmp, sizex);
    for (size_t i = 0; i < 26; i++) {
        before_hot_data1[i] = 0; after_hot_data1[i] = 0;
        before_hot_data2[i] = 0; after_hot_data2[i] = 0;
    }

    // Kernel arguments setup
    uint64_t args1[7];
    args1[0] = (uint64_t)nx;
    args1[1] = (uint64_t)ny;
    args1[2] = (uint64_t)A;
    args1[3] = (uint64_t)x;
    args1[4] = (uint64_t)tmp;
    args1[5] = (uint64_t)before_hot_data1;
    args1[6] = (uint64_t)after_hot_data1;

    uint64_t args2[7];
    args2[0] = (uint64_t)nx;
    args2[1] = (uint64_t)ny;
    args2[2] = (uint64_t)A;
    args2[3] = (uint64_t)y_dsp;
    args2[4] = (uint64_t)tmp;
    args2[5] = (uint64_t)before_hot_data2;
    args2[6] = (uint64_t)after_hot_data2;

    // Device execution with separate timing
    int groupId = hthread_group_create(clusterId, nthreads);
    if (groupId == -1) { fprintf(stderr, "Failed to create group with %d threads\n", nthreads); return 2; }

    uint64_t start, end;

    // DSP execution for kernel1
    start = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel1, 2, 5, args1);
    hthread_group_wait(groupId);
    end = getCurrentTimeMicros();
    timeDsp1 = (end - start);

    // DSP execution for kernel2
    start = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel2, 2, 5, args2);
    hthread_group_wait(groupId);
    end = getCurrentTimeMicros();
    timeDsp2 = (end - start);

    hthread_group_destroy(groupId);

    // Host execution with separate timing for each kernel
    start = getCurrentTimeMicros();
    atax_cpu1(nx, ny, A_host, x_host, tmp_host);
    end = getCurrentTimeMicros();
    timeCpu1 = (end - start);

    start = getCurrentTimeMicros();
    atax_cpu2(nx, ny, A_host, y_host, tmp_host);
    end = getCurrentTimeMicros();
    timeCpu2 = (end - start);

    // Result validation
    int errNum = check_result(ny, y_host, y_dsp);
    if (errNum != 0) {
        printf("Failed to test ATax!\n");
    } else {
        save_data(nx, ny, before_hot_data1, after_hot_data1, timeDsp1, timeCpu1, clusterId, devProgram, nthreads, kernel1);
        save_data(nx, ny, before_hot_data2, after_hot_data2, timeDsp2, timeCpu2, clusterId, devProgram, nthreads, kernel2);
        printf("WallTime ATAX_kernel1 (DSP/CPU): %fs / %fs\n", timeDsp1 / 1e6, timeCpu1 / 1e6);
        printf("WallTime ATAX_kernel2 (DSP/CPU): %fs / %fs\n", timeDsp2 / 1e6, timeCpu2 / 1e6);
    }

    // Resource cleanup
    hthread_free(A);
    hthread_free(x);
    hthread_free(y_dsp);
    hthread_free(tmp);
    hthread_free(before_hot_data1); hthread_free(after_hot_data1);
    hthread_free(before_hot_data2); hthread_free(after_hot_data2);
    free(A_host);
    free(x_host);
    free(y_host);
    free(tmp_host);
    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return errNum != 0 ? 1 : 0;
}