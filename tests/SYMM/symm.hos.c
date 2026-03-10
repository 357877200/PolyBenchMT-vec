#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

#include "hthread_host.h"
#include "../common/tool.h"

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/*************************************
 * CPU 分解后的两个 kernel
 ************************************/
// 第一阶段：更新 C[k,j] 和计算 temp2
void symm_cpu1(int m, int n, double alpha,
               double *A, double *B, double *C,
               double *temp2_arr)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double temp2 = 0.0;
            for (int k = 0; k < i; k++) {
                C[k * n + j] += alpha * B[i * n + j] * A[i * m + k];
                temp2 += B[k * n + j] * A[i * m + k];
            }
            temp2_arr[i * n + j] = temp2;
        }
    }
}

// 第二阶段：根据 temp2 更新 C[i,j]
void symm_cpu2(int m, int n, double alpha, double beta,
               double *A, double *B, double *C,
               double *temp2_arr)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double temp2 = temp2_arr[i * n + j];
            C[i * n + j] = beta * C[i * n + j]
                         + alpha * B[i * n + j] * A[i * m + i]
                         + alpha * temp2;
        }
    }
}

/*************************************
 * 初始化数据
 *************************************/
void init_arrays(int m, int n, double *alpha, double *beta,
                 double *A, double *B, double *C)
{
    *alpha = 1.5;
    *beta  = 1.2;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = ((i + j) % 100) / (double)m;
            B[i * n + j] = ((n + i - j) % 100) / (double)m;
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j <= i; j++) { // lower-triangle
            A[i * m + j] = ((i + j) % 100) / (double)m;
        }
        for (int j = i + 1; j < m; j++) {
            A[i * m + j] = -999.0; // unused region
        }
    }
}

/*************************************
 * 校验结果
 *************************************/
int check_result(int m, int n, double *C_host, double *C_dev)
{
    int errNum = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (percentDiff(C_host[i * n + j], C_dev[i * n + j])
                 > PERCENT_DIFF_ERROR_THRESHOLD) {
                if (errNum < 10) {
                    fprintf(stderr, "C diff @[%d][%d]: H=%.6f D=%.6f\n",
                            i, j, C_host[i * n + j], C_dev[i * n + j]);
                }
                errNum++;
            }
        }
    }
    if (errNum) {
        double fail_percent = (100.0 * errNum) / (double)(m * n);
        fprintf(stderr,
            "Non-Matching CPU-DSP Outputs: %d (%.2lf%%)\n",
            errNum, fail_percent);
    }
    return errNum;
}

/*************************************
 * 保存性能数据
 *************************************/
void save_data(int m, int n,
               uint64_t *before_hot_data, uint64_t *after_hot_data,
               uint64_t timeDsp, uint64_t timeCpu,
               int clusterId, char *devProgram, int nthreads, char *kernel)
{
    FILE *file = fopen("tests/SYMM/symm_events.txt", "a");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }
    fprintf(file, "%d,%d,%s,%s,%d,", clusterId, m * n, devProgram, kernel, nthreads);
    for (int eid = 0; eid < 26; eid++) {
        fprintf(file, "%lu", after_hot_data[eid] - before_hot_data[eid]);
        if (eid < 25) fprintf(file, ",");
    }
    fprintf(file, ",%fs,%fs\n", timeDsp / 1e6, timeCpu / 1e6);
    fclose(file);
}

/*************************************
 * 主程序
 *************************************/
int main(int argc, char **argv)
{
    int retc;
    int clusterId = 1;
    int m = 1000;
    int n = 1000;
    int nthreads = 1;
    char *devProgram = "operators/SYMM/symm.dev.dat";
    char *kernel1 = "symm_kernel1";
    char *kernel2 = "symm_kernel2";
    uint64_t timeDsp1 = 0, timeDsp2 = 0;
    uint64_t timeCpu1 = 0, timeCpu2 = 0;

    // 命令行解析
    for (int i = 1; i < argc; i++) {
        if (i + 1 < argc) {
            if (strcmp(argv[i], "--clusterId") == 0 || strcmp(argv[i], "-c") == 0) {
                clusterId = atoi(argv[i + 1]); i++;
            } else if (strcmp(argv[i], "--m") == 0) {
                m = atoi(argv[i + 1]); i++;
            } else if (strcmp(argv[i], "--n") == 0) {
                n = atoi(argv[i + 1]); i++;
            } else if (strcmp(argv[i], "--program") == 0 || strcmp(argv[i], "-p") == 0) {
                devProgram = argv[i + 1]; i++;
            } else if (strcmp(argv[i], "--threads") == 0 || strcmp(argv[i], "-t") == 0) {
                nthreads = atoi(argv[i + 1]); i++;
            } else if (strcmp(argv[i], "--kernel1") == 0 || strcmp(argv[i], "-k1") == 0) {
                kernel1 = argv[i + 1]; i++;
            } else if (strcmp(argv[i], "--kernel2") == 0 || strcmp(argv[i], "-k2") == 0) {
                kernel2 = argv[i + 1]; i++;
            }
        }
    }

    if (clusterId < 0 || clusterId > 3) { fprintf(stderr, "invalid clusterId\n"); return 2; }
    if (nthreads <= 0) { fprintf(stderr, "invalid nthreads\n"); return 2; }
    if (access(devProgram, F_OK) != 0) { fprintf(stderr, "%s not found\n", devProgram); return 2; }

    retc = hthread_dev_open(clusterId); if (retc != 0) { fprintf(stderr, "Failed to open device\n"); return retc; }
    retc = hthread_dat_load(clusterId, devProgram); if (retc != 0) { fprintf(stderr, "Failed to load program\n"); return retc; }
    int availThreads = hthread_get_avail_threads(clusterId);
    if (nthreads > availThreads) {
        fprintf(stderr, "threads overflow: avail %d, ask %d\n", availThreads, nthreads);
        hthread_dat_unload(clusterId);
        hthread_dev_close(clusterId);
        return 2;
    }

    size_t sizeA = (size_t)m * m * sizeof(double);
    size_t sizeB = (size_t)m * n * sizeof(double);
    size_t sizeC = (size_t)m * n * sizeof(double);
    size_t sizeTemp2 = (size_t)m * n * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    // 设备端内存
    double *A_d = (double *)hthread_malloc(clusterId, sizeA, HT_MEM_RO);
    double *B_d = (double *)hthread_malloc(clusterId, sizeB, HT_MEM_RO);
    double *C_d = (double *)hthread_malloc(clusterId, sizeC, HT_MEM_RW);
    double *temp2_d = (double *)hthread_malloc(clusterId, sizeTemp2, HT_MEM_RW);
    uint64_t *before_hot_data1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after_hot_data1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before_hot_data2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after_hot_data2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    if (!A_d || !B_d || !C_d || !temp2_d || !before_hot_data1 || !after_hot_data1 || !before_hot_data2 || !after_hot_data2) {
        fprintf(stderr, "device malloc failed\n"); return 1;
    }

    // 主机端内存
    double *A_h = (double *)malloc(sizeA);
    double *B_h = (double *)malloc(sizeB);
    double *C_h = (double *)malloc(sizeC);
    double *temp2_h = (double *)malloc(sizeTemp2);
    if (!A_h || !B_h || !C_h || !temp2_h) { fprintf(stderr, "malloc host fail\n"); return 1; }

    double alpha, beta;
    init_arrays(m, n, &alpha, &beta, A_h, B_h, C_h);

    memcpy(A_d, A_h, sizeA);
    memcpy(B_d, B_h, sizeB);
    memcpy(C_d, C_h, sizeC);
    memset(temp2_d, 0, sizeTemp2);
    memset(temp2_h, 0, sizeTemp2);
    memset(before_hot_data1, 0, sizeHot);
    memset(after_hot_data1, 0, sizeHot);
    memset(before_hot_data2, 0, sizeHot);
    memset(after_hot_data2, 0, sizeHot);

    // kernel1 参数
    uint64_t args1[9];
    args1[0] = (uint64_t)m;
    args1[1] = (uint64_t)n;
    args1[2] = doubleToRawBits(alpha);
    args1[3] = (uint64_t)A_d;
    args1[4] = (uint64_t)B_d;
    args1[5] = (uint64_t)C_d;
    args1[6] = (uint64_t)temp2_d;
    args1[7] = (uint64_t)before_hot_data1;
    args1[8] = (uint64_t)after_hot_data1;

    // kernel2 参数
    uint64_t args2[10];
    args2[0] = (uint64_t)m;
    args2[1] = (uint64_t)n;
    args2[2] = doubleToRawBits(alpha);
    args2[3] = doubleToRawBits(beta);
    args2[4] = (uint64_t)A_d;
    args2[5] = (uint64_t)B_d;
    args2[6] = (uint64_t)C_d;
    args2[7] = (uint64_t)temp2_d;
    args2[8] = (uint64_t)before_hot_data2;
    args2[9] = (uint64_t)after_hot_data2;

    // 设备端执行 kernel1
    int groupId = hthread_group_create(clusterId, nthreads);
    if (groupId == -1) { fprintf(stderr, "group create fail\n"); return 2; }
    uint64_t start, end;
    start = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel1, 3, 6, args1);
    hthread_group_wait(groupId);
    end = getCurrentTimeMicros();
    timeDsp1 = end - start;

    // 设备端执行 kernel2
    start = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel2, 4, 6, args2);
    hthread_group_wait(groupId);
    end = getCurrentTimeMicros();
    timeDsp2 = end - start;
    hthread_group_destroy(groupId);

    // 主机端执行
    start = getCurrentTimeMicros();
    symm_cpu1(m, n, alpha, A_h, B_h, C_h, temp2_h);
    end = getCurrentTimeMicros();
    timeCpu1 = end - start;

    start = getCurrentTimeMicros();
    symm_cpu2(m, n, alpha, beta, A_h, B_h, C_h, temp2_h);
    end = getCurrentTimeMicros();
    timeCpu2 = end - start;

    // 校验结果
    int errNum = check_result(m, n, C_h, C_d);
    if (errNum != 0) {
        fprintf(stderr, "SYMM test FAILED!\n");
    } else {
        save_data(m, n, before_hot_data1, after_hot_data1, timeDsp1, timeCpu1, clusterId, devProgram, nthreads, kernel1);
        save_data(m, n, before_hot_data2, after_hot_data2, timeDsp2, timeCpu2, clusterId, devProgram, nthreads, kernel2);
        printf("WallTime SYMM_kernel1 (DSP/CPU): %fs / %fs\n", timeDsp1 / 1e6, timeCpu1 / 1e6);
        printf("WallTime SYMM_kernel2 (DSP/CPU): %fs / %fs\n", timeDsp2 / 1e6, timeCpu2 / 1e6);
    }

    // 释放资源
    hthread_free(A_d);
    hthread_free(B_d);
    hthread_free(C_d);
    hthread_free(temp2_d);
    hthread_free(before_hot_data1); hthread_free(after_hot_data1);
    hthread_free(before_hot_data2); hthread_free(after_hot_data2);
    free(A_h); free(B_h); free(C_h); free(temp2_h);
    hthread_dat_unload(clusterId); hthread_dev_close(clusterId);

    return errNum != 0 ? 1 : 0;
}