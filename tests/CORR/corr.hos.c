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
#define FLOAT_N 3214212.01f
#define EPS 0.005f

#define sqrt_of_array_cell(x, j) sqrt(x[j])

//==================== CPU Version ====================//

// Step1: 计算每列均值
void correlation_cpu1_mean(int m, int n, double *mean, double *data) {
    for (int j = 0; j < m; j++) {
        mean[j] = 0.0;
        for (int i = 0; i < n; i++) {
            mean[j] += data[i * m + j];
        }
        mean[j] /= (double)FLOAT_N;
    }
}

// Step2: 计算标准差
void correlation_cpu2_stddev(int m, int n, double *mean, double *stddev, double *data) {
    for (int j = 0; j < m; j++) {
        stddev[j] = 0.0;
        for (int i = 0; i < n; i++) {
            stddev[j] += (data[i * m + j] - mean[j]) * (data[i * m + j] - mean[j]);
        }
        stddev[j] /= FLOAT_N;
        stddev[j] = sqrt(stddev[j]);
        stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
    }
}

// Step3: 数据标准化
void correlation_cpu3_normalize(int m, int n, double *mean, double *stddev, double *data) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            data[i * m + j] -= mean[j];
            data[i * m + j] /= (sqrt(FLOAT_N) * stddev[j]);
        }
    }
}

// Step4: 计算相关系数矩阵
void correlation_cpu4_corrmat(int m, int n, double *symmat, double *data) {
    // 计算相关系数矩阵
    for (int j1 = 0; j1 < m - 1; j1++) {
        symmat[j1 * m + j1] = 1.0;
        for (int j2 = j1 + 1; j2 < m; j2++) {
            symmat[j1 * m + j2] = 0.0;
            for (int i = 0; i < n; i++) {
                symmat[j1 * m + j2] += (data[i * m + j1] * data[i * m + j2]);
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
    FILE *file = fopen("tests/CORR/corr_events.txt", "a");
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
    char *devProgram = "operators/CORR/corr.dev.dat";
    int nthreads = 1;
    char *kernel1 = "corr_kernel1";
    char *kernel2 = "corr_kernel2";
    char *kernel3 = "corr_kernel3";
    char *kernel4 = "corr_kernel4";

    // 计时变量
    uint64_t timeDsp1 = 0, timeDsp2 = 0, timeDsp3 = 0, timeDsp4 = 0;
    uint64_t timeCpu1 = 0, timeCpu2 = 0, timeCpu3 = 0, timeCpu4 = 0;

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
            else if (strcmp(argv[i], "--kernel4") == 0 || strcmp(argv[i], "-k4") == 0) kernel4 = argv[++i];
        }
    }

    // DSP 初始化
    if (hthread_dev_open(clusterId) != 0) { fprintf(stderr, "Failed to open device\n"); return 1; }
    if (hthread_dat_load(clusterId, devProgram) != 0) { fprintf(stderr, "Failed to load program\n"); return 1; }
    int availThreads = hthread_get_avail_threads(clusterId);
    if (nthreads > availThreads) { fprintf(stderr, "Too many threads: %d > %d\n", nthreads, availThreads); return 1; }

    // 内存大小
    size_t sizeData   = (size_t)n * m * sizeof(double);
    size_t sizeSym    = (size_t)m * m * sizeof(double);
    size_t sizeMean   = (size_t)m * sizeof(double);
    size_t sizeStddev = (size_t)m * sizeof(double);
    size_t sizeHot    = 26 * sizeof(uint64_t);

    // 分配 DSP 内存
    double *data_d    = (double *)hthread_malloc(clusterId, sizeData, HT_MEM_RW);
    double *symmat_d  = (double *)hthread_malloc(clusterId, sizeSym, HT_MEM_RW);
    double *mean_d    = (double *)hthread_malloc(clusterId, sizeMean, HT_MEM_RW);
    double *stddev_d  = (double *)hthread_malloc(clusterId, sizeStddev, HT_MEM_RW);

    uint64_t *before_hot1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after_hot1  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before_hot2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after_hot2  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before_hot3 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after_hot3  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before_hot4 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after_hot4  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);

    // 分配 CPU 内存
    double *data_h        = (double *)malloc(sizeData);
    double *data_backup   = (double *)malloc(sizeData);
    double *symmat_h      = (double *)malloc(sizeSym);
    double *mean_h        = (double *)malloc(sizeMean);
    double *stddev_h      = (double *)malloc(sizeStddev);
    double *symmat_from_d = (double *)malloc(sizeSym);

    // 初始化数据
    init_array(m, n, data_h);
    memcpy(data_backup, data_h, sizeData);
    memcpy(data_d, data_h, sizeData);
    memset(symmat_d, 0, sizeSym);
    memset(before_hot1, 0, sizeHot);
    memset(after_hot1, 0, sizeHot);
    memset(before_hot2, 0, sizeHot);
    memset(after_hot2, 0, sizeHot);
    memset(before_hot3, 0, sizeHot);
    memset(after_hot3, 0, sizeHot);
    memset(before_hot4, 0, sizeHot);
    memset(after_hot4, 0, sizeHot);

    // 内核参数
    uint64_t args1[6] = { m, n, (uint64_t)mean_d, (uint64_t)data_d, (uint64_t)before_hot1, (uint64_t)after_hot1 };
    uint64_t args2[7] = { m, n, (uint64_t)mean_d, (uint64_t)stddev_d, (uint64_t)data_d, (uint64_t)before_hot2, (uint64_t)after_hot2 };
    uint64_t args3[7] = { m, n, (uint64_t)mean_d, (uint64_t)stddev_d, (uint64_t)data_d, (uint64_t)before_hot3, (uint64_t)after_hot3 };
    uint64_t args4[6] = { m, n, (uint64_t)symmat_d, (uint64_t)data_d, (uint64_t)before_hot4, (uint64_t)after_hot4 };

    // DSP 执行
    int gid = hthread_group_create(clusterId, nthreads);
    uint64_t start, end;

    // Step 1: 计算均值
    start = getCurrentTimeMicros();
    hthread_group_exec(gid, kernel1, 2, 4, args1);
    hthread_group_wait(gid);
    end = getCurrentTimeMicros();
    timeDsp1 = end - start;

    // Step 2: 计算标准差
    start = getCurrentTimeMicros();
    hthread_group_exec(gid, kernel2, 2, 5, args2);
    hthread_group_wait(gid);
    end = getCurrentTimeMicros();
    timeDsp2 = end - start;

    // Step 3: 数据标准化
    start = getCurrentTimeMicros();
    hthread_group_exec(gid, kernel3, 2, 5, args3);
    hthread_group_wait(gid);
    end = getCurrentTimeMicros();
    timeDsp3 = end - start;

    // Step 4: 计算相关系数矩阵
    start = getCurrentTimeMicros();
    hthread_group_exec(gid, kernel4, 2, 4, args4);
    hthread_group_wait(gid);
    end = getCurrentTimeMicros();
    timeDsp4 = end - start;

    hthread_group_destroy(gid);
    memcpy(symmat_from_d, symmat_d, sizeSym);

    // CPU 执行，逐步计时
    memcpy(data_h, data_backup, sizeData);

    start = getCurrentTimeMicros();
    correlation_cpu1_mean(m, n, mean_h, data_h);
    end = getCurrentTimeMicros();
    timeCpu1 = end - start;

    start = getCurrentTimeMicros();
    correlation_cpu2_stddev(m, n, mean_h, stddev_h, data_h);
    end = getCurrentTimeMicros();
    timeCpu2 = end - start;

    start = getCurrentTimeMicros();
    correlation_cpu3_normalize(m, n, mean_h, stddev_h, data_h);
    end = getCurrentTimeMicros();
    timeCpu3 = end - start;

    start = getCurrentTimeMicros();
    correlation_cpu4_corrmat(m, n, symmat_h, data_h);
    end = getCurrentTimeMicros();
    timeCpu4 = end - start;

    // 结果比较
    int errNum = check_result(m, symmat_h, symmat_from_d);
    if (errNum == 0) {
        save_data(m, n, before_hot1, after_hot1, timeDsp1, timeCpu1, clusterId, devProgram, nthreads, kernel1);
        save_data(m, n, before_hot2, after_hot2, timeDsp2, timeCpu2, clusterId, devProgram, nthreads, kernel2);
        save_data(m, n, before_hot3, after_hot3, timeDsp3, timeCpu3, clusterId, devProgram, nthreads, kernel3);
        save_data(m, n, before_hot4, after_hot4, timeDsp4, timeCpu4, clusterId, devProgram, nthreads, kernel4);

        printf("WallTime corr_kernel1 (DSP/CPU): %fs / %fs\n", timeDsp1 / 1e6, timeCpu1 / 1e6);
        printf("WallTime corr_kernel2 (DSP/CPU): %fs / %fs\n", timeDsp2 / 1e6, timeCpu2 / 1e6);
        printf("WallTime corr_kernel3 (DSP/CPU): %fs / %fs\n", timeDsp3 / 1e6, timeCpu3 / 1e6);
        printf("WallTime corr_kernel4 (DSP/CPU): %fs / %fs\n", timeDsp4 / 1e6, timeCpu4 / 1e6);
    } else {
        fprintf(stderr, "CORR result verification FAILED!\n");
    }

    // 释放资源
    hthread_free(data_d); hthread_free(symmat_d); hthread_free(mean_d); hthread_free(stddev_d);
    hthread_free(before_hot1); hthread_free(after_hot1);
    hthread_free(before_hot2); hthread_free(after_hot2);
    hthread_free(before_hot3); hthread_free(after_hot3);
    hthread_free(before_hot4); hthread_free(after_hot4);

    free(data_h); free(data_backup); free(symmat_h); free(mean_h); free(stddev_h); free(symmat_from_d);

    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return errNum ? 1 : 0;
}