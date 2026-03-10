// author：cjk  time：2025/7/30
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include "hthread_host.h"
#include "../common/tool.h"

void mm2_cpu1(int ni, int nj, int nk, int nl, double alpha, double beta, double *tmp, double *A, double *B, double *C, double *D) {
    int i, j, k;
    // First matrix multiplication: tmp = alpha * A * B
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            tmp[i * nj + j] = 0.0;
            for (k = 0; k < nk; k++) {
                tmp[i * nj + j] += alpha * A[i * nk + k] * B[k * nj + j];
            }
        }
    }
}
void mm2_cpu2(int ni, int nj, int nk, int nl, double alpha, double beta, double *tmp, double *A, double *B, double *C, double *D) {
    int i, j, k;
    // Second matrix multiplication: D = tmp * C + beta * D
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nl; j++) {
            D[i * nl + j] *= beta;
            for (k = 0; k < nj; k++) {
                D[i * nl + j] += tmp[i * nj + k] * C[k * nl + j];
            }
        }
    }
}

int check_result(int ni, int nl, double *D_host, double *D) {
    int errNum = 0;
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            int idx = i * nl + j;
            if (fabs(D[idx] - D_host[idx]) > 1e-5) {
                if (errNum < 10) {
                    fprintf(stderr, "Data error at (%d,%d): Host=%.2f, Device=%.2f\n",
                            i, j, D_host[idx], D[idx]);
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
int check_tmp_result(int ni, int nj, double *tmp_host, double *tmp) {
    int errNum = 0;
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            int idx = i * nj + j;
            if (fabs(tmp[idx] - tmp_host[idx]) > 1e-5) {
                if (errNum < 10) {
                    fprintf(stderr, "Data error at (%d,%d): Host=%.2f, Device=%.2f\n",
                            i, j, tmp_host[idx], tmp[idx]);
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

void save_data(int ni, int nl, uint64_t *before_hot_data, uint64_t *after_hot_data, uint64_t timeDsp, uint64_t timeCpu, int clusterId, char *devProgram, int nthreads,char *kernel) {
    FILE *file = fopen("tests/2MM/2mm_events.txt", "a");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }
    fprintf(file, "%d,%d,%s,%s,%d,Difference,", clusterId, ni * nl, devProgram,kernel, nthreads);
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
    int ni = 4096;
    int nj = 4096;
    int nk = 4096;
    int nl = 4096;
    double alpha = 1.5;
    double beta = 1.2;
    char *devProgram = "operators/2MM/2mm.dev.dat";
    int nthreads = 1;
    char *kernel1 = "mm2_kernel1";
    char *kernel2 = "mm2_kernel2";
    uint64_t timeGold1 = 0, timeDev1 = 0,timeGold2 = 0, timeDev2 = 0;

    // 解析命令行参数（保持不变）
    for (int i = 1; i < argc; i++) {
        if (i + 1 < argc) {
            if (strcmp(argv[i], "--clusterId") == 0 || strcmp(argv[i], "-c") == 0) {
                clusterId = atoi(argv[i + 1]);
                i++;
            } else if (strcmp(argv[i], "--ni") == 0) {
                ni = atoi(argv[i + 1]);
                i++;
            } else if (strcmp(argv[i], "--nj") == 0) {
                nj = atoi(argv[i + 1]);
                i++;
            } else if (strcmp(argv[i], "--nk") == 0) {
                nk = atoi(argv[i + 1]);
                i++;
            } else if (strcmp(argv[i], "--nl") == 0) {
                nl = atoi(argv[i + 1]);
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

    // 设备初始化和参数检查（保持不变）
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

    // 重点：内存空间分配，根据矩阵尺寸精确分配
    size_t sizeA = (size_t)ni * nk * sizeof(double);
    size_t sizeB = (size_t)nk * nj * sizeof(double);
    size_t sizeC = (size_t)nj * nl * sizeof(double);
    size_t sizeD = (size_t)ni * nl * sizeof(double);
    size_t sizeTmp = (size_t)ni * nj * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    // 设备内存分配
    double *A = (double *)hthread_malloc(clusterId, sizeA, HT_MEM_RO); if (!A) { fprintf(stderr, "Malloc A failed\n"); return 1; }
    double *B = (double *)hthread_malloc(clusterId, sizeB, HT_MEM_RO); if (!B) { fprintf(stderr, "Malloc B failed\n"); return 1; }
    double *C = (double *)hthread_malloc(clusterId, sizeC, HT_MEM_RO); if (!C) { fprintf(stderr, "Malloc C failed\n"); return 1; }
    double *D = (double *)hthread_malloc(clusterId, sizeD, HT_MEM_RW); if (!D) { fprintf(stderr, "Malloc D failed\n"); return 1; }
    double *tmp = (double *)hthread_malloc(clusterId, sizeTmp, HT_MEM_RW); if (!tmp) { fprintf(stderr, "Malloc tmp failed\n"); return 1; }

    // 主机内存分配 (用于验证)
    double *A_host = (double *)malloc(sizeA); if (!A_host) { fprintf(stderr, "Malloc A_host failed\n"); return 1; }
    double *B_host = (double *)malloc(sizeB); if (!B_host) { fprintf(stderr, "Malloc B_host failed\n"); return 1; }
    double *C_host = (double *)malloc(sizeC); if (!C_host) { fprintf(stderr, "Malloc C_host failed\n"); return 1; }
    double *D_host = (double *)malloc(sizeD); if (!D_host) { fprintf(stderr, "Malloc D_host failed\n"); return 1; }
    double *tmp_host = (double *)malloc(sizeTmp); if (!tmp_host) { fprintf(stderr, "Malloc tmp_host failed\n"); return 1; }


    // 事件计数器内存分配 (合并为一对before/after，简化处理)
    uint64_t *before_hot_data1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!before_hot_data1) { return 1; }
    uint64_t *after_hot_data1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!after_hot_data1) { return 1; }
    uint64_t *before_hot_data2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!before_hot_data2) { return 1; }
    uint64_t *after_hot_data2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!after_hot_data2) { return 1; }

    // 数据初始化
    for (size_t i = 0; i < ni * nk; i++) A[i] = A_host[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    for (size_t i = 0; i < nk * nj; i++) B[i] = B_host[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    for (size_t i = 0; i < nj * nl; i++) C[i] = C_host[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    for (size_t i = 0; i < ni * nl; i++) D[i] = D_host[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    for (size_t i = 0; i < ni * nj; i++) tmp[i] = tmp_host[i] = 0.0;
    for (size_t i = 0; i < 26; i++) { before_hot_data1[i] = 0; after_hot_data1[i] = 0;  before_hot_data2[i] = 0; after_hot_data2[i] = 0; }

    // 第一个核函数参数 (tmp = alpha * A * B)
    uint64_t args1[9];
    args1[0] = (uint64_t)ni;
    args1[1] = (uint64_t)nj;
    args1[2] = (uint64_t)nk;
    args1[3] = (uint64_t)doubleToRawBits(alpha);
    args1[4] = (uint64_t)tmp;
    args1[5] = (uint64_t)A;
    args1[6] = (uint64_t)B;
    args1[7] = (uint64_t)before_hot_data1;
    args1[8] = (uint64_t)after_hot_data1;

    // 第二个核函数参数 (D = tmp * C + beta * D)
    uint64_t args2[9];
    args2[0] = (uint64_t)ni;
    args2[1] = (uint64_t)nj;
    args2[2] = (uint64_t)nl;
    args2[3] = (uint64_t)doubleToRawBits(beta);
    args2[4] = (uint64_t)tmp;
    args2[5] = (uint64_t)C;
    args2[6] = (uint64_t)D;
    args2[7] = (uint64_t)before_hot_data2;
    args2[7] = (uint64_t)after_hot_data2;

    // 设备端执行
    timeDev1 = getCurrentTimeMicros();
    int threadId1 = hthread_group_create(clusterId, nthreads, kernel1, 4, 9, args1);
    if (threadId1 == -1) { fprintf(stderr, "Failed to create threads with %s\n", kernel1); return 2; }
    hthread_group_wait(threadId1);
    timeDev1 = getCurrentTimeMicros() - timeDev1;
    
    timeDev2 = getCurrentTimeMicros();
    int threadId2 = hthread_group_create(clusterId, nthreads, kernel2, 4, 9, args2);
    if (threadId2 == -1) { fprintf(stderr, "Failed to create threads with %s\n", kernel2); return 2; }
    hthread_group_wait(threadId2);
    timeDev2 = getCurrentTimeMicros() - timeDev2;

    // 主机端执行
    timeGold1 = getCurrentTimeMicros();
    mm2_cpu1(ni, nj, nk, nl, alpha, beta, tmp_host, A_host, B_host, C_host, D_host);
    timeGold1 = getCurrentTimeMicros() - timeGold1;

    timeGold2 = getCurrentTimeMicros();
    mm2_cpu2(ni, nj, nk, nl, alpha, beta, tmp_host, A_host, B_host, C_host, D_host);
    timeGold2 = getCurrentTimeMicros() - timeGold2;

    // 结果验证
    int errNum = check_result(ni, nl, D_host, D);
    // check_tmp_result(ni, nj, tmp_host, tmp);
    if (errNum != 0) {
        fprintf(stderr, "Failed to test 2MM!\n");
    } else {
        save_data(ni, nl, before_hot_data1, after_hot_data1, timeDev1, timeGold1, clusterId, devProgram, nthreads,kernel1);
        save_data(ni, nl, before_hot_data1, after_hot_data1, timeDev2, timeGold2, clusterId, devProgram, nthreads,kernel2);
        printf("WallTime 2MM_kernel1 (DSP/CPU): %fs / %fs\n", timeDev1 / 1e6, timeGold1 / 1e6);
        printf("WallTime 2MM_kernel2 (DSP/CPU): %fs / %fs\n", timeDev2 / 1e6, timeGold2 / 1e6);
    }

    // 资源释放
    hthread_free(A); hthread_free(B); hthread_free(C); hthread_free(D); hthread_free(tmp);
    hthread_free(before_hot_data1); hthread_free(after_hot_data1);
    hthread_free(before_hot_data2); hthread_free(after_hot_data2);
    free(A_host); free(B_host); free(C_host); free(D_host); free(tmp_host); 
    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);
    return errNum != 0 ? 1 : 0;
}