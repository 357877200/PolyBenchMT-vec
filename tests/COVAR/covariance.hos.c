#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include "hthread_host.h"
#include "../common/tool.h"

#define PERCENT_DIFF_ERROR_THRESHOLD 1.05
#define FLOAT_N 3214212.01

//==================== CPU Version ====================//

// Step1: 计算每列均值
void covariance_cpu1_mean(int m, int n, double *mean, double *data) {
    for (int j = 0; j < m; j++) {
        mean[j] = 0.0;
        for (int i = 0; i < n; i++) {
            mean[j] += data[i * m + j];
        }
        mean[j] /= (double)n;
    }
}

// Step2: 数据中心化
void covariance_cpu2_reduce(int m, int n, double *mean, double *data) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            data[i * m + j] -= mean[j];
        }
    }
}

// Step3: 协方差矩阵
void covariance_cpu3_covmat(int m, int n, double *symmat, double *data) {
    for (int j1 = 0; j1 < m; j1++) {
        for (int j2 = j1; j2 < m; j2++) {
            symmat[j1 * m + j2] = 0.0;
            for (int i = 0; i < n; i++) {
                symmat[j1 * m + j2] += data[i * m + j1] * data[i * m + j2];
            }
            symmat[j2 * m + j1] = symmat[j1 * m + j2];
        }
    }
}

// 初始化数据
void init_array(int m, int n, double *data) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            data[i * m + j] = ((double)i * j) / m;
        }
    }
}

// 检查结果
int check_result(int m, double *symmat_host, double *symmat_dsp) {
    int errNum = 0;
    for (int i = 0; i < m * m; i++) {
        if (percentDiff(symmat_host[i], symmat_dsp[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
            if (errNum < 10) {
                fprintf(stderr, "Error at [%d]: Host=%.6f, Device=%.6f\n", i, symmat_host[i], symmat_dsp[i]);
            }
            errNum++;
        }
    }
    if (errNum > 0) {
        fprintf(stderr, "Total errors: %d\n", errNum);
    }
    return errNum;
}

// 保存性能数据
void save_data(int m, int n, uint64_t *before_hot, uint64_t *after_hot,
               uint64_t timeDsp, uint64_t timeCpu,
               int clusterId, char *devProgram, int nthreads, char *kernel) {
    FILE *file = fopen("tests/COVAR/covar_events.txt", "a");
    if (!file) { perror("open file"); return; }
    fprintf(file, "%d,%d,%s,%s,%d,Difference,", clusterId, m * n, devProgram, kernel, nthreads);
    for (int eid = 0; eid < 26; eid++) {
        fprintf(file, "%lu", after_hot[eid] - before_hot[eid]);
        if (eid < 25) fprintf(file, ",");
    }
    fprintf(file, ",%fs,%fs\n", timeDsp / 1e6, timeCpu / 1e6);
    fclose(file);
}

//==================== Main ====================//
int main(int argc, char **argv) {
    int clusterId = 0;
    int m = 128;   // 列数
    int n = 128;   // 行数
    char *devProgram = "operators/COVAR/covariance.dev.dat";
    int nthreads = 1;
    char *kernel1 = "covar_kernel1";
    char *kernel2 = "covar_kernel2";
    char *kernel3 = "covar_kernel3";

    // 计时变量
    uint64_t timeDsp1 = 0, timeDsp2 = 0, timeDsp3 = 0;
    uint64_t timeCpu1 = 0, timeCpu2 = 0, timeCpu3 = 0;

    // 命令行参数解析
    for (int i = 1; i < argc; i++) {
        if (i + 1 < argc) {
            if (strcmp(argv[i], "--clusterId") == 0 || strcmp(argv[i], "-c") == 0) clusterId = atoi(argv[++i]);
            else if (strcmp(argv[i], "--m") == 0) m = atoi(argv[++i]);
            else if (strcmp(argv[i], "--n") == 0) n = atoi(argv[++i]);
            else if (strcmp(argv[i], "--program") == 0 || strcmp(argv[i], "-p") == 0) devProgram = argv[++i];
            else if (strcmp(argv[i], "--threads") == 0 || strcmp(argv[i], "-t") == 0) nthreads = atoi(argv[++i]);
            else if (strcmp(argv[i], "--kernel1") == 0 || strcmp(argv[i], "-k1") == 0) kernel1 = argv[++i];
            else if (strcmp(argv[i], "--kernel2") == 0 || strcmp(argv[i], "-k2") == 0) kernel2 = argv[++i];
            else if (strcmp(argv[i], "--kernel3") == 0 || strcmp(argv[i], "-k3") == 0) kernel3 = argv[++i];
        }
    }

    // DSP 初始化
    if (hthread_dev_open(clusterId) != 0) { fprintf(stderr, "Failed to open device\n"); return 1; }
    if (hthread_dat_load(clusterId, devProgram) != 0) { fprintf(stderr, "Failed to load program\n"); return 1; }
    int availThreads = hthread_get_avail_threads(clusterId);
    if (nthreads > availThreads) { fprintf(stderr, "Too many threads: %d > %d\n", nthreads, availThreads); return 1; }

    // 内存大小
    size_t sizeData  = (size_t)n * m * sizeof(double);
    size_t sizeSym   = (size_t)m * m * sizeof(double);
    size_t sizeMean  = (size_t)m * sizeof(double);
    size_t sizeHot   = 26 * sizeof(uint64_t);

    // 分配 DSP 内存
    double *data_d   = (double *)hthread_malloc(clusterId, sizeData, HT_MEM_RW);
    double *symmat_d = (double *)hthread_malloc(clusterId, sizeSym, HT_MEM_RW);
    double *mean_d   = (double *)hthread_malloc(clusterId, sizeMean, HT_MEM_RW);

    uint64_t *before_hot1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after_hot1  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before_hot2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after_hot2  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before_hot3 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after_hot3  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);

    // 分配 CPU 内存
    double *data_h        = (double *)malloc(sizeData);
    double *symmat_h      = (double *)malloc(sizeSym);
    double *mean_h        = (double *)malloc(sizeMean);
    double *symmat_from_d = (double *)malloc(sizeSym);

    // 初始化数据
    init_array(m, n, data_h);
    memcpy(data_d, data_h, sizeData);
    memset(symmat_d, 0, sizeSym);
    memset(before_hot1, 0, sizeHot);
    memset(after_hot1, 0, sizeHot);
    memset(before_hot2, 0, sizeHot);
    memset(after_hot2, 0, sizeHot);
    memset(before_hot3, 0, sizeHot);
    memset(after_hot3, 0, sizeHot);

    // 内核参数
    uint64_t args1[6] = { m, n, (uint64_t)mean_d, (uint64_t)data_d, (uint64_t)before_hot1, (uint64_t)after_hot1 };
    uint64_t args2[6] = { m, n, (uint64_t)mean_d, (uint64_t)data_d, (uint64_t)before_hot2, (uint64_t)after_hot2 };
    uint64_t args3[6] = { m, n, (uint64_t)symmat_d, (uint64_t)data_d, (uint64_t)before_hot3, (uint64_t)after_hot3 };

    // DSP 执行
    int gid = hthread_group_create(clusterId, nthreads);
    uint64_t start, end;

    start = getCurrentTimeMicros();
    hthread_group_exec(gid, kernel1, 2, 4, args1);
    hthread_group_wait(gid);
    end = getCurrentTimeMicros();
    timeDsp1 = end - start;

    start = getCurrentTimeMicros();
    hthread_group_exec(gid, kernel2, 2, 4, args2);
    hthread_group_wait(gid);
    end = getCurrentTimeMicros();
    timeDsp2 = end - start;

    start = getCurrentTimeMicros();
    hthread_group_exec(gid, kernel3, 2, 4, args3);
    hthread_group_wait(gid);
    end = getCurrentTimeMicros();
    timeDsp3 = end - start;

    hthread_group_destroy(gid);
    memcpy(symmat_from_d, symmat_d, sizeSym);

    // CPU 执行，逐步计时
    memcpy(data_h, data_d, sizeData);

    start = getCurrentTimeMicros();
    covariance_cpu1_mean(m, n, mean_h, data_h);
    end = getCurrentTimeMicros();
    timeCpu1 = end - start;

    start = getCurrentTimeMicros();
    covariance_cpu2_reduce(m, n, mean_h, data_h);
    end = getCurrentTimeMicros();
    timeCpu2 = end - start;

    start = getCurrentTimeMicros();
    covariance_cpu3_covmat(m, n, symmat_h, data_h);
    end = getCurrentTimeMicros();
    timeCpu3 = end - start;

    // 结果比较
    int errNum = check_result(m, symmat_h, symmat_from_d);
    if (errNum == 0) {
        save_data(m, n, before_hot1, after_hot1, timeDsp1, timeCpu1, clusterId, devProgram, nthreads, kernel1);
        save_data(m, n, before_hot2, after_hot2, timeDsp2, timeCpu2, clusterId, devProgram, nthreads, kernel2);
        save_data(m, n, before_hot3, after_hot3, timeDsp3, timeCpu3, clusterId, devProgram, nthreads, kernel3);

        printf("WallTime covar_kernel1 (DSP/CPU): %fs / %fs\n", timeDsp1 / 1e6, timeCpu1 / 1e6);
        printf("WallTime covar_kernel2 (DSP/CPU): %fs / %fs\n", timeDsp2 / 1e6, timeCpu2 / 1e6);
        printf("WallTime covar_kernel3 (DSP/CPU): %fs / %fs\n", timeDsp3 / 1e6, timeCpu3 / 1e6);
    } else {
        fprintf(stderr, "Covariance result verification FAILED!\n");
    }

    // 释放资源
    hthread_free(data_d); hthread_free(symmat_d); hthread_free(mean_d);
    hthread_free(before_hot1); hthread_free(after_hot1);
    hthread_free(before_hot2); hthread_free(after_hot2);
    hthread_free(before_hot3); hthread_free(after_hot3);

    free(data_h); free(symmat_h); free(mean_h); free(symmat_from_d);

    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return errNum ? 1 : 0;
}