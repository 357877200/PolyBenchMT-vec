// author: adapted from polybench ADI  time: 2025/8/20
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include "hthread_host.h"
#include "../common/tool.h"
#define PERCENT_DIFF_ERROR_THRESHOLD 2.5
// CPU implementations for each kernel
void adi_cpu1(int n, double *A, double *B, double *X) {
    for (int i1 = 0; i1 < n; i1++) {
        for (int i2 = 1; i2 < n; i2++) {
            X[i1 * n + i2] = X[i1 * n + i2] - X[i1 * n + (i2 - 1)] * A[i1 * n + i2] / B[i1 * n + (i2 - 1)];
            B[i1 * n + i2] = B[i1 * n + i2] - A[i1 * n + i2] * A[i1 * n + i2] / B[i1 * n + (i2 - 1)];
        }
    }
}

void adi_cpu2(int n, double *A, double *B, double *X) {
    for (int i1 = 0; i1 < n; i1++) {
        X[i1 * n + (n - 1)] = X[i1 * n + (n - 1)] / B[i1 * n + (n - 1)];
    }
}

void adi_cpu3(int n, double *A, double *B, double *X) {
    for (int i1 = 0; i1 < n; i1++) {
        for (int i2 = 0; i2 < n - 2; i2++) {
            X[i1 * n + (n - i2 - 2)] = (X[i1 * n + (n - 2 - i2)] - X[i1 * n + (n - 2 - i2 - 1)] * A[i1 * n + (n - i2 - 3)]) / B[i1 * n + (n - 3 - i2)];
        }
    }
}

void adi_cpu4(int n, int i1, double *A, double *B, double *X) {
    for (int i2 = 0; i2 < n; i2++) {
        X[i1 * n + i2] = X[i1 * n + i2] - X[(i1 - 1) * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
        B[i1 * n + i2] = B[i1 * n + i2] - A[i1 * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
    }
}

void adi_cpu5(int n, double *A, double *B, double *X) {
    for (int i2 = 0; i2 < n; i2++) {
        X[(n - 1) * n + i2] = X[(n - 1) * n + i2] / B[(n - 1) * n + i2];
    }
}

void adi_cpu6(int n, int i1, double *A, double *B, double *X) {
    for (int i2 = 0; i2 < n; i2++) {
        X[(n - 2 - i1) * n + i2] = (X[(n - 2 - i1) * n + i2] - X[(n - i1 - 3) * n + i2] * A[(n - 3 - i1) * n + i2]) / B[(n - 2 - i1) * n + i2];
    }
}

// Initialize input arrays
void init_array(int n, double *A, double *B, double *X) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            X[i * n + j] = ((double)i * (j + 1) + 1) / n;
            A[i * n + j] = ((double)(i - 1) * (j + 4) + 2) / n;
            B[i * n + j] = ((double)(i + 3) * (j + 7) + 3) / n;
        }
    }
}

// Check results between CPU and DSP
int check_result(int n, double *B_host, double *B_device, double *X_host, double *X_device) {
    int errNum = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            if (percentDiff(B_host[idx], B_device[idx]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                if (errNum < 10) {
                    fprintf(stderr, "B data error at (%d,%d): Host=%.2f, Device=%.2f\n", i, j, B_host[idx], B_device[idx]);
                }
                errNum++;
            }
            if (percentDiff(X_host[idx], X_device[idx]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                if (errNum < 10) {
                    fprintf(stderr, "X data error at (%d,%d): Host=%.2f, Device=%.2f\n", i, j, X_host[idx], X_device[idx]);
                }
                errNum++;
            }
        }
    }
    if (errNum != 0) {
        fprintf(stderr, "Total errors: %d\n", errNum);
    }
    return errNum;
}

// Save performance data
void save_data(int n, uint64_t *before_hot_data, uint64_t *after_hot_data, uint64_t timeDsp, uint64_t timeCpu, int clusterId, char *devProgram, int nthreads, char *kernel) {
    FILE *file = fopen("tests/ADI/adi_events.txt", "a");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }
    fprintf(file, "%d,%d,%s,%s,%d,Difference,", clusterId, n * n, devProgram, kernel, nthreads);
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
    int tsteps = 10;  // Default tsteps
    int n = 64;       // Default problem size
    char *devProgram = "operators/ADI/adi.dev.dat";
    int nthreads = 1;
    char *kernel1 = "adi_kernel1";
    char *kernel2 = "adi_kernel2";
    char *kernel3 = "adi_kernel3";
    char *kernel4 = "adi_kernel4";
    char *kernel5 = "adi_kernel5";
    char *kernel6 = "adi_kernel6";
    uint64_t timeDsp1 = 0, timeDsp2 = 0, timeDsp3 = 0, timeDsp4 = 0, timeDsp5 = 0, timeDsp6 = 0;
    uint64_t timeCpu1 = 0, timeCpu2 = 0, timeCpu3 = 0, timeCpu4 = 0, timeCpu5 = 0, timeCpu6 = 0;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (i + 1 < argc) {
            if (strcmp(argv[i], "--clusterId") == 0 || strcmp(argv[i], "-c") == 0) {
                clusterId = atoi(argv[i + 1]);
                i++;
            } else if (strcmp(argv[i], "--tsteps") == 0) {
                tsteps = atoi(argv[i + 1]);
                i++;
            } else if (strcmp(argv[i], "--n") == 0) {
                n = atoi(argv[i + 1]);
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
            } else if (strcmp(argv[i], "--kernel3") == 0 || strcmp(argv[i], "-k3") == 0) {
                kernel3 = argv[i + 1];
                i++;
            } else if (strcmp(argv[i], "--kernel4") == 0 || strcmp(argv[i], "-k4") == 0) {
                kernel4 = argv[i + 1];
                i++;
            } else if (strcmp(argv[i], "--kernel5") == 0 || strcmp(argv[i], "-k5") == 0) {
                kernel5 = argv[i + 1];
                i++;
            } else if (strcmp(argv[i], "--kernel6") == 0 || strcmp(argv[i], "-k6") == 0) {
                kernel6 = argv[i + 1];
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
    size_t sizeArray = (size_t)n * n * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    // Device memory allocation
    double *A = (double *)hthread_malloc(clusterId, sizeArray, HT_MEM_RO); if (!A) { fprintf(stderr, "Malloc A failed\n"); return 1; }
    double *B = (double *)hthread_malloc(clusterId, sizeArray, HT_MEM_RW); if (!B) { fprintf(stderr, "Malloc B failed\n"); return 1; }
    double *X = (double *)hthread_malloc(clusterId, sizeArray, HT_MEM_RW); if (!X) { fprintf(stderr, "Malloc X failed\n"); return 1; }
    double *B_dsp = (double *)hthread_malloc(clusterId, sizeArray, HT_MEM_RW); if (!B_dsp) { fprintf(stderr, "Malloc B_dsp failed\n"); return 1; }
    double *X_dsp = (double *)hthread_malloc(clusterId, sizeArray, HT_MEM_RW); if (!X_dsp) { fprintf(stderr, "Malloc X_dsp failed\n"); return 1; }

    // Host memory allocation (for validation)
    double *A_host = (double *)malloc(sizeArray); if (!A_host) { fprintf(stderr, "Malloc A_host failed\n"); return 1; }
    double *B_host = (double *)malloc(sizeArray); if (!B_host) { fprintf(stderr, "Malloc B_host failed\n"); return 1; }
    double *X_host = (double *)malloc(sizeArray); if (!X_host) { fprintf(stderr, "Malloc X_host failed\n"); return 1; }

    // Event counter memory allocation for each kernel
    uint64_t *before_hot_data1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!before_hot_data1) { return 1; }
    uint64_t *after_hot_data1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!after_hot_data1) { return 1; }
    uint64_t *before_hot_data2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!before_hot_data2) { return 1; }
    uint64_t *after_hot_data2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!after_hot_data2) { return 1; }
    uint64_t *before_hot_data3 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!before_hot_data3) { return 1; }
    uint64_t *after_hot_data3 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!after_hot_data3) { return 1; }
    uint64_t *before_hot_data4 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!before_hot_data4) { return 1; }
    uint64_t *after_hot_data4 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!after_hot_data4) { return 1; }
    uint64_t *before_hot_data5 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!before_hot_data5) { return 1; }
    uint64_t *after_hot_data5 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!after_hot_data5) { return 1; }
    uint64_t *before_hot_data6 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!before_hot_data6) { return 1; }
    uint64_t *after_hot_data6 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!after_hot_data6) { return 1; }

    // Data initialization
    init_array(n, A, B, X);
    memcpy(A_host, A, sizeArray);
    memcpy(B_host, B, sizeArray);
    memcpy(X_host, X, sizeArray);
    memcpy(B_dsp, B, sizeArray);
    memcpy(X_dsp, X, sizeArray);
    for (size_t i = 0; i < 26; i++) {
        before_hot_data1[i] = 0; after_hot_data1[i] = 0;
        before_hot_data2[i] = 0; after_hot_data2[i] = 0;
        before_hot_data3[i] = 0; after_hot_data3[i] = 0;
        before_hot_data4[i] = 0; after_hot_data4[i] = 0;
        before_hot_data5[i] = 0; after_hot_data5[i] = 0;
        before_hot_data6[i] = 0; after_hot_data6[i] = 0;
    }

    // Kernel arguments setup based on __global__ function signatures
    uint64_t args1[6];
    args1[0] = (uint64_t)n;
    args1[1] = (uint64_t)A;
    args1[2] = (uint64_t)B_dsp;
    args1[3] = (uint64_t)X_dsp;
    args1[4] = (uint64_t)before_hot_data1;
    args1[5] = (uint64_t)after_hot_data1;

    uint64_t args2[6];
    args2[0] = (uint64_t)n;
    args2[1] = (uint64_t)A;
    args2[2] = (uint64_t)B_dsp;
    args2[3] = (uint64_t)X_dsp;
    args2[4] = (uint64_t)before_hot_data2;
    args2[5] = (uint64_t)after_hot_data2;

    uint64_t args3[6];
    args3[0] = (uint64_t)n;
    args3[1] = (uint64_t)A;
    args3[2] = (uint64_t)B_dsp;
    args3[3] = (uint64_t)X_dsp;
    args3[4] = (uint64_t)before_hot_data3;
    args3[5] = (uint64_t)after_hot_data3;

    uint64_t args4[7];
    args4[0] = (uint64_t)n;
    args4[1] = (uint64_t)0;  // i1, to be set in loop
    args4[2] = (uint64_t)A;
    args4[3] = (uint64_t)B_dsp;
    args4[4] = (uint64_t)X_dsp;
    args4[5] = (uint64_t)before_hot_data4;
    args4[6] = (uint64_t)after_hot_data4;

    uint64_t args5[6];
    args5[0] = (uint64_t)n;
    args5[1] = (uint64_t)A;
    args5[2] = (uint64_t)B_dsp;
    args5[3] = (uint64_t)X_dsp;
    args5[4] = (uint64_t)before_hot_data5;
    args5[5] = (uint64_t)after_hot_data5;

    uint64_t args6[7];
    args6[0] = (uint64_t)n;
    args6[1] = (uint64_t)0;  // i1, to be set in loop
    args6[2] = (uint64_t)A;
    args6[3] = (uint64_t)B_dsp;
    args6[4] = (uint64_t)X_dsp;
    args6[5] = (uint64_t)before_hot_data6;
    args6[6] = (uint64_t)after_hot_data6;

    // Device execution with separate timing
    int groupId = hthread_group_create(clusterId, nthreads);
    if (groupId == -1) { fprintf(stderr, "Failed to create group with %d threads\n", nthreads); return 2; }

    for (int t = 0; t < tsteps; t++) {
        uint64_t start, end;

        start = getCurrentTimeMicros();
        hthread_group_exec(groupId, kernel1, 1, 6, args1);
        hthread_group_wait(groupId);
        end = getCurrentTimeMicros();
        timeDsp1 += (end - start);

        start = getCurrentTimeMicros();
        hthread_group_exec(groupId, kernel2, 1, 6, args2);
        hthread_group_wait(groupId);
        end = getCurrentTimeMicros();
        timeDsp2 += (end - start);

        start = getCurrentTimeMicros();
        hthread_group_exec(groupId, kernel3, 1, 6, args3);
        hthread_group_wait(groupId);
        end = getCurrentTimeMicros();
        timeDsp3 += (end - start);

        for (int i1 = 1; i1 < n; i1++) {
            args4[1] = (uint64_t)i1;
            start = getCurrentTimeMicros();
            hthread_group_exec(groupId, kernel4, 2, 7, args4);
            hthread_group_wait(groupId);
            end = getCurrentTimeMicros();
            timeDsp4 += (end - start);
        }

        start = getCurrentTimeMicros();
        hthread_group_exec(groupId, kernel5, 1, 6, args5);
        hthread_group_wait(groupId);
        end = getCurrentTimeMicros();
        timeDsp5 += (end - start);

        for (int i1 = 0; i1 < n - 2; i1++) {
            args6[1] = (uint64_t)i1;
            start = getCurrentTimeMicros();
            hthread_group_exec(groupId, kernel6, 2, 7, args6);
            hthread_group_wait(groupId);
            end = getCurrentTimeMicros();
            timeDsp6 += (end - start);
        }
    }
    hthread_group_destroy(groupId);

    // Host execution with separate timing for each kernel
    for (int t = 0; t < tsteps; t++) {
        uint64_t start, end;

        start = getCurrentTimeMicros();
        adi_cpu1(n, A_host, B_host, X_host);
        end = getCurrentTimeMicros();
        timeCpu1 += (end - start);

        start = getCurrentTimeMicros();
        adi_cpu2(n, A_host, B_host, X_host);
        end = getCurrentTimeMicros();
        timeCpu2 += (end - start);

        start = getCurrentTimeMicros();
        adi_cpu3(n, A_host, B_host, X_host);
        end = getCurrentTimeMicros();
        timeCpu3 += (end - start);

        for (int i1 = 1; i1 < n; i1++) {
            start = getCurrentTimeMicros();
            adi_cpu4(n, i1, A_host, B_host, X_host);
            end = getCurrentTimeMicros();
            timeCpu4 += (end - start);
        }

        start = getCurrentTimeMicros();
        adi_cpu5(n, A_host, B_host, X_host);
        end = getCurrentTimeMicros();
        timeCpu5 += (end - start);

        for (int i1 = 0; i1 < n - 2; i1++) {
            start = getCurrentTimeMicros();
            adi_cpu6(n, i1, A_host, B_host, X_host);
            end = getCurrentTimeMicros();
            timeCpu6 += (end - start);
        }
    }

    // Result validation
    int errNum = check_result(n, B_host, B_dsp, X_host, X_dsp);
    if (errNum != 0) {
        printf("Failed to test ADI!\n");
    } else {
        save_data(n, before_hot_data1, after_hot_data1, timeDsp1, timeCpu1, clusterId, devProgram, nthreads, kernel1);
        save_data(n, before_hot_data2, after_hot_data2, timeDsp2, timeCpu2, clusterId, devProgram, nthreads, kernel2);
        save_data(n, before_hot_data3, after_hot_data3, timeDsp3, timeCpu3, clusterId, devProgram, nthreads, kernel3);
        save_data(n, before_hot_data4, after_hot_data4, timeDsp4, timeCpu4, clusterId, devProgram, nthreads, kernel4);
        save_data(n, before_hot_data5, after_hot_data5, timeDsp5, timeCpu5, clusterId, devProgram, nthreads, kernel5);
        save_data(n, before_hot_data6, after_hot_data6, timeDsp6, timeCpu6, clusterId, devProgram, nthreads, kernel6);
        printf("WallTime ADI_kernel1 (DSP/CPU): %fs / %fs\n", timeDsp1 / 1e6, timeCpu1 / 1e6);
        printf("WallTime ADI_kernel2 (DSP/CPU): %fs / %fs\n", timeDsp2 / 1e6, timeCpu2 / 1e6);
        printf("WallTime ADI_kernel3 (DSP/CPU): %fs / %fs\n", timeDsp3 / 1e6, timeCpu3 / 1e6);
        printf("WallTime ADI_kernel4 (DSP/CPU): %fs / %fs\n", timeDsp4 / 1e6, timeCpu4 / 1e6);
        printf("WallTime ADI_kernel5 (DSP/CPU): %fs / %fs\n", timeDsp5 / 1e6, timeCpu5 / 1e6);
        printf("WallTime ADI_kernel6 (DSP/CPU): %fs / %fs\n", timeDsp6 / 1e6, timeCpu6 / 1e6);
    }

    // Resource cleanup
    hthread_free(A);
    hthread_free(B);
    hthread_free(X);
    hthread_free(B_dsp);
    hthread_free(X_dsp);
    hthread_free(before_hot_data1); hthread_free(after_hot_data1);
    hthread_free(before_hot_data2); hthread_free(after_hot_data2);
    hthread_free(before_hot_data3); hthread_free(after_hot_data3);
    hthread_free(before_hot_data4); hthread_free(after_hot_data4);
    hthread_free(before_hot_data5); hthread_free(after_hot_data5);
    hthread_free(before_hot_data6); hthread_free(after_hot_data6);
    free(A_host);
    free(B_host);
    free(X_host);
    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return errNum != 0 ? 1 : 0;
}