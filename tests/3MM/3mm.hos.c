// author: cjk  time: 2025/8/20
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include "hthread_host.h"
#include "../common/tool.h"

void mm3_cpu1(int ni, int nj, int nk, double *E, double *A, double *B) {
    int i, j, k;
    // First matrix multiplication: E = A * B
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            E[i * nj + j] = 0.0;
            for (k = 0; k < nk; k++) {
                E[i * nj + j] += A[i * nk + k] * B[k * nj + j];
            }
        }
    }
}

void mm3_cpu2(int nj, int nl, int nm, double *F, double *C, double *D) {
    int i, j, k;
    // Second matrix multiplication: F = C * D
    for (i = 0; i < nj; i++) {
        for (j = 0; j < nl; j++) {
            F[i * nl + j] = 0.0;
            for (k = 0; k < nm; k++) {
                F[i * nl + j] += C[i * nm + k] * D[k * nl + j];
            }
        }
    }
}

void mm3_cpu3(int ni, int nl, int nj, double *G, double *E, double *F) {
    int i, j, k;
    // Third matrix multiplication: G = E * F
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nl; j++) {
            G[i * nl + j] = 0.0;
            for (k = 0; k < nj; k++) {
                G[i * nl + j] += E[i * nj + k] * F[k * nl + j];
            }
        }
    }
}

int check_result(int ni, int nl, double *G_host, double *G) {
    int errNum = 0;
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            int idx = i * nl + j;
            if (fabs(G[idx] - G_host[idx]) > 1e-5) {
                if (errNum < 10) {
                    fprintf(stderr, "Data error at (%d,%d): Host=%.2f, Device=%.2f\n",
                            i, j, G_host[idx], G[idx]);
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

void save_data(int ni, int nl, uint64_t *before_hot_data, uint64_t *after_hot_data, uint64_t timeDsp, uint64_t timeCpu, int clusterId, char *devProgram, int nthreads, char *kernel) {
    FILE *file = fopen("tests/3MM/3mm_events.txt", "a");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }
    fprintf(file, "%d,%d,%s,%s,%d,Difference,", clusterId, ni * nl, devProgram, kernel, nthreads);
    for (int eid = 0; eid < 26; eid++) {
        fprintf(file, "%lu", after_hot_data[eid] - before_hot_data[eid]);
        if (eid < 25) fprintf(file, ",");
    }
    fprintf(file, ",%fs,%fs\n", timeDsp / 1e6, timeCpu / 1e6);
    fclose(file);
}

int main(int argc, char **argv) {
    int retc;
    int clusterId = 0;
    int ni = 64;
    int nj = 64;
    int nk = 64;
    int nl = 64;
    int nm = 64;
    char *devProgram = "operators/3MM/3mm.dev.dat";
    int nthreads = 1;
    char *kernel1 = "mm3_kernel1";
    char *kernel2 = "mm3_kernel2";
    char *kernel3 = "mm3_kernel3";
    uint64_t timeGold1 = 0, timeDev1 = 0, timeGold2 = 0, timeDev2 = 0, timeGold3 = 0, timeDev3 = 0;

    // 解析命令行参数
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
            } else if (strcmp(argv[i], "--nm") == 0) {
                nm = atoi(argv[i + 1]);
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
            }
        }
    }

    // 设备初始化和参数检查
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

    // 内存空间分配，根据矩阵尺寸精确分配
    size_t sizeA = (size_t)ni * nk * sizeof(double);
    size_t sizeB = (size_t)nk * nj * sizeof(double);
    size_t sizeC = (size_t)nj * nm * sizeof(double);
    size_t sizeD = (size_t)nm * nl * sizeof(double);
    size_t sizeE = (size_t)ni * nj * sizeof(double);
    size_t sizeF = (size_t)nj * nl * sizeof(double);
    size_t sizeG = (size_t)ni * nl * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    // 设备内存分配
    double *A = (double *)hthread_malloc(clusterId, sizeA, HT_MEM_RO); if (!A) { fprintf(stderr, "Malloc A failed\n"); return 1; }
    double *B = (double *)hthread_malloc(clusterId, sizeB, HT_MEM_RO); if (!B) { fprintf(stderr, "Malloc B failed\n"); return 1; }
    double *C = (double *)hthread_malloc(clusterId, sizeC, HT_MEM_RO); if (!C) { fprintf(stderr, "Malloc C failed\n"); return 1; }
    double *D = (double *)hthread_malloc(clusterId, sizeD, HT_MEM_RO); if (!D) { fprintf(stderr, "Malloc D failed\n"); return 1; }
    double *E = (double *)hthread_malloc(clusterId, sizeE, HT_MEM_RW); if (!E) { fprintf(stderr, "Malloc E failed\n"); return 1; }
    double *F = (double *)hthread_malloc(clusterId, sizeF, HT_MEM_RW); if (!F) { fprintf(stderr, "Malloc F failed\n"); return 1; }
    double *G = (double *)hthread_malloc(clusterId, sizeG, HT_MEM_RW); if (!G) { fprintf(stderr, "Malloc G failed\n"); return 1; }

    // 主机内存分配 (用于验证)
    double *A_host = (double *)malloc(sizeA); if (!A_host) { fprintf(stderr, "Malloc A_host failed\n"); return 1; }
    double *B_host = (double *)malloc(sizeB); if (!B_host) { fprintf(stderr, "Malloc B_host failed\n"); return 1; }
    double *C_host = (double *)malloc(sizeC); if (!C_host) { fprintf(stderr, "Malloc C_host failed\n"); return 1; }
    double *D_host = (double *)malloc(sizeD); if (!D_host) { fprintf(stderr, "Malloc D_host failed\n"); return 1; }
    double *E_host = (double *)malloc(sizeE); if (!E_host) { fprintf(stderr, "Malloc E_host failed\n"); return 1; }
    double *F_host = (double *)malloc(sizeF); if (!F_host) { fprintf(stderr, "Malloc F_host failed\n"); return 1; }
    double *G_host = (double *)malloc(sizeG); if (!G_host) { fprintf(stderr, "Malloc G_host failed\n"); return 1; }

    // 事件计数器内存分配
    uint64_t *before_hot_data1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!before_hot_data1) { return 1; }
    uint64_t *after_hot_data1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!after_hot_data1) { return 1; }
    uint64_t *before_hot_data2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!before_hot_data2) { return 1; }
    uint64_t *after_hot_data2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!after_hot_data2) { return 1; }
    uint64_t *before_hot_data3 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!before_hot_data3) { return 1; }
    uint64_t *after_hot_data3 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW); if (!after_hot_data3) { return 1; }

    // 数据初始化
    for (size_t i = 0; i < ni * nk; i++) A[i] = A_host[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    for (size_t i = 0; i < nk * nj; i++) B[i] = B_host[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    for (size_t i = 0; i < nj * nm; i++) C[i] = C_host[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    for (size_t i = 0; i < nm * nl; i++) D[i] = D_host[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    for (size_t i = 0; i < 26; i++) {
        before_hot_data1[i] = 0; after_hot_data1[i] = 0;
        before_hot_data2[i] = 0; after_hot_data2[i] = 0;
        before_hot_data3[i] = 0; after_hot_data3[i] = 0;
    }

    // 核函数参数设置 - 根据提供的核函数签名调整
    // mm3_kernel1(int ni, int nj, int nk, int nl, int nm, double *A, double *B, double *E)
    uint64_t args1[10];
    args1[0] = (uint64_t)ni;
    args1[1] = (uint64_t)nj;
    args1[2] = (uint64_t)nk;
    args1[3] = (uint64_t)nl;
    args1[4] = (uint64_t)nm;
    args1[5] = (uint64_t)A;
    args1[6] = (uint64_t)B;
    args1[7] = (uint64_t)E;
    args1[8] = (uint64_t)before_hot_data1;
    args1[9] = (uint64_t)after_hot_data1;

    // mm3_kernel2(int ni, int nj, int nk, int nl, int nm, double *C, double *D, double *F)
    uint64_t args2[10];
    args2[0] = (uint64_t)ni;
    args2[1] = (uint64_t)nj;
    args2[2] = (uint64_t)nk;
    args2[3] = (uint64_t)nl;
    args2[4] = (uint64_t)nm;
    args2[5] = (uint64_t)C;
    args2[6] = (uint64_t)D;
    args2[7] = (uint64_t)F;
    args2[8] = (uint64_t)before_hot_data2;
    args2[9] = (uint64_t)after_hot_data2;

    // mm3_kernel3(int ni, int nj, int nk, int nl, int nm, double *E, double *F, double *G)
    uint64_t args3[10];
    args3[0] = (uint64_t)ni;
    args3[1] = (uint64_t)nj;
    args3[2] = (uint64_t)nk;
    args3[3] = (uint64_t)nl;
    args3[4] = (uint64_t)nm;
    args3[5] = (uint64_t)E;
    args3[6] = (uint64_t)F;
    args3[7] = (uint64_t)G;
    args3[8] = (uint64_t)before_hot_data3;
    args3[9] = (uint64_t)after_hot_data3;

    // 设备端执行
    timeDev1 = getCurrentTimeMicros();
    int threadId1 = hthread_group_create(clusterId, nthreads, kernel1, 5, 10, args1); // 调整参数数量
    if (threadId1 == -1) { fprintf(stderr, "Failed to create threads with %s\n", kernel1); return 2; }
    hthread_group_wait(threadId1);
    timeDev1 = getCurrentTimeMicros() - timeDev1;

    timeDev2 = getCurrentTimeMicros();
    int threadId2 = hthread_group_create(clusterId, nthreads, kernel2, 5, 10, args2); // 调整参数数量
    if (threadId2 == -1) { fprintf(stderr, "Failed to create threads with %s\n", kernel2); return 2; }
    hthread_group_wait(threadId2);
    timeDev2 = getCurrentTimeMicros() - timeDev2;

    timeDev3 = getCurrentTimeMicros();
    int threadId3 = hthread_group_create(clusterId, nthreads, kernel3, 5, 10, args3); // 调整参数数量
    if (threadId3 == -1) { fprintf(stderr, "Failed to create threads with %s\n", kernel3); return 2; }
    hthread_group_wait(threadId3);
    timeDev3 = getCurrentTimeMicros() - timeDev3;

    // 主机端执行
    timeGold1 = getCurrentTimeMicros();
    mm3_cpu1(ni, nj, nk, E_host, A_host, B_host);
    timeGold1 = getCurrentTimeMicros() - timeGold1;

    timeGold2 = getCurrentTimeMicros();
    mm3_cpu2(nj, nl, nm, F_host, C_host, D_host);
    timeGold2 = getCurrentTimeMicros() - timeGold2;

    timeGold3 = getCurrentTimeMicros();
    mm3_cpu3(ni, nl, nj, G_host, E_host, F_host);
    timeGold3 = getCurrentTimeMicros() - timeGold3;

    // 结果验证
    int errNum = check_result(ni, nl, G_host, G);
    if (errNum != 0) {
        printf("Failed to test 3MM!\n");
    } else {
        save_data(ni, nl, before_hot_data1, after_hot_data1, timeDev1, timeGold1, clusterId, devProgram, nthreads, kernel1);
        save_data(ni, nl, before_hot_data2, after_hot_data2, timeDev2, timeGold2, clusterId, devProgram, nthreads, kernel2);
        save_data(ni, nl, before_hot_data3, after_hot_data3, timeDev3, timeGold3, clusterId, devProgram, nthreads, kernel3);

        printf("WallTime 3MM_kernel1 (DSP/CPU): %fs / %fs\n", timeDev1 / 1e6, timeGold1 / 1e6);
        printf("WallTime 3MM_kernel2 (DSP/CPU): %fs / %fs\n", timeDev2 / 1e6, timeGold2 / 1e6);
        printf("WallTime 3MM_kernel3 (DSP/CPU): %fs / %fs\n", timeDev3 / 1e6, timeGold3 / 1e6);
    }

    // 资源释放
    hthread_free(A); hthread_free(B); hthread_free(C); hthread_free(D); hthread_free(E); hthread_free(F); hthread_free(G);
    hthread_free(before_hot_data1); hthread_free(after_hot_data1);
    hthread_free(before_hot_data2); hthread_free(after_hot_data2);
    hthread_free(before_hot_data3); hthread_free(after_hot_data3);
    free(A_host); free(B_host); free(C_host); free(D_host); free(E_host); free(F_host); free(G_host);
    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);
    return errNum != 0 ? 1 : 0;
}