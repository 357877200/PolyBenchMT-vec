#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "hthread_host.h"
#include "../common/tool.h"

#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

// 初始化数据
void init_array(int n, double *r) {
    for (int i = 0; i < n; i++) {
        r[i] = (double)(n + 1 - i);
    }
}

// CPU版本 kernel1：计算本轮迭代的 sum、alpha、临时z
void durbin_cpu1(int k, int n, double *r, double *y, double *z, double *alpha, double *beta) {
    double sum = 0.0;
    *beta = (1.0 - (*alpha) * (*alpha)) * (*beta);
    for (int i = 0; i < k; i++) {
        sum += r[k - i - 1] * y[i];
    }
    *alpha = -(r[k] + sum) / (*beta);
    for (int i = 0; i < k; i++) {
        z[i] = y[i] + (*alpha) * y[k - i - 1];
    }
}

// CPU版本 kernel2：更新 y
void durbin_cpu2(int k, double *y, double *z, double alpha) {
    for (int i = 0; i < k; i++) {
        y[i] = z[i];
    }
    y[k] = alpha;
}

// 比较结果
int check_result(int n, double *y_host, double *y_device) {
    int errNum = 0;
    for (int i = 0; i < n; i++) {
        if (percentDiff(y_host[i], y_device[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
            if (errNum < 10) {
                fprintf(stderr, "y error at %d: Host=%.6f, Device=%.6f\n", i, y_host[i], y_device[i]);
            }
            errNum++;
        }
    }
    if (errNum > 0) fprintf(stderr, "Total errors: %d\n", errNum);
    return errNum;
}

void save_data(int n, uint64_t *before_hot, uint64_t *after_hot, uint64_t timeDsp, uint64_t timeCpu,
               int clusterId, char *devProgram, int nthreads, char *kernel) {
    FILE *file = fopen("tests/DURBIN/durbin_events.txt", "a");
    if (!file) { perror("open durbin_events.txt"); return; }
    fprintf(file, "%d,%d,%s,%s,%d,", clusterId, n, devProgram, kernel, nthreads);
    for (int i = 0; i < 26; i++) {
        fprintf(file, "%lu", after_hot[i] - before_hot[i]);
        if (i < 25) fprintf(file, ",");
    }
    fprintf(file, ",%fs,%fs\n", timeDsp / 1e6, timeCpu / 1e6);
    fclose(file);
}

int main(int argc, char **argv) {
    int clusterId = 1;
    int n = 4096;  // 默认规模
    char *devProgram = "operators/DURBIN/durbin.dev.dat";
    int nthreads = 1;
    char *kernel1 = "durbin_kernel1";
    char *kernel2 = "durbin_kernel2";

// 命令行参数解析（支持长参数和缩写）
for (int i = 1; i < argc; i++) {
    if (i + 1 < argc) {
        if (strcmp(argv[i], "--clusterId") == 0 || strcmp(argv[i], "-c") == 0) {
            clusterId = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--n") == 0) {
            n = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--program") == 0 || strcmp(argv[i], "-p") == 0) {
            devProgram = argv[++i];
        }
        else if (strcmp(argv[i], "--threads") == 0 || strcmp(argv[i], "-t") == 0) {
            nthreads = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--kernel1") == 0 || strcmp(argv[i], "-k1") == 0) {
            kernel1 = argv[++i];
        }
        else if (strcmp(argv[i], "--kernel2") == 0 || strcmp(argv[i], "-k2") == 0) {
            kernel2 = argv[++i];
        }
    }
}

    if (clusterId < 0 || clusterId > 3) { fprintf(stderr, "invalid clusterId\n"); return 2; }
    if (nthreads <= 0) { fprintf(stderr, "invalid nthreads\n"); return 2; }
    if (access(devProgram, F_OK) != 0) { fprintf(stderr, "program file not found\n"); return 2; }

    if (hthread_dev_open(clusterId) != 0) { fprintf(stderr, "Failed open device\n"); return 1; }
    if (hthread_dat_load(clusterId, devProgram) != 0) { fprintf(stderr, "Failed load program\n"); return 1; }
    if (nthreads > hthread_get_avail_threads(clusterId)) { fprintf(stderr, "Too many threads\n"); return 2; }

    // 分配内存
    size_t sizeArr = n * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    double *r = (double *)hthread_malloc(clusterId, sizeArr, HT_MEM_RO);
    double *y_dsp = (double *)hthread_malloc(clusterId, sizeArr, HT_MEM_RW);
    double *z = (double *)hthread_malloc(clusterId, sizeArr, HT_MEM_RW);
    double *alpha_dsp = (double *)hthread_malloc(clusterId, sizeof(double), HT_MEM_RW);
    double *beta_dsp = (double *)hthread_malloc(clusterId, sizeof(double), HT_MEM_RW);

    double *r_host = (double *)malloc(sizeArr);
    double *y_host = (double *)malloc(sizeArr);
    double *z_host = (double *)malloc(sizeArr);
    double alpha_host, beta_host;

    uint64_t *before1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after1  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after2  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);

    // 初始化
    init_array(n, r);
    memcpy(r_host, r, sizeArr);
    memset(y_dsp, 0, sizeArr);
    memset(z, 0, sizeArr);
    memset(y_host, 0, sizeArr);
    alpha_host = -r_host[0];
    beta_host = 1.0;
    y_host[0] = -r_host[0];
    memcpy(alpha_dsp, &alpha_host, sizeof(double));
    memcpy(beta_dsp, &beta_host, sizeof(double));
    memcpy(y_dsp, y_host, sizeof(double));

    memset(before1, 0, sizeHot);
    memset(after1, 0, sizeHot);
    memset(before2, 0, sizeHot);
    memset(after2, 0, sizeHot);
    int barrier_id = hthread_barrier_create(clusterId);
// ---------- 参数 ----------
uint64_t args1[9], args2[6];
args1[0] = 0;                 
args1[1] = (uint64_t)barrier_id;     
args1[2] = (uint64_t)n; 
args1[3] = (uint64_t)r;
args1[4] = (uint64_t)y_dsp;
args1[5] = (uint64_t)z;
args1[6] = (uint64_t)alpha_dsp;
args1[7] = (uint64_t)beta_dsp;
args1[8] = (uint64_t)before1;
args1[9] = (uint64_t)after1;

// kernel2 参数
args2[0] = 0; // 每轮动态赋
args2[1] = (uint64_t)barrier_id; // 每轮动态赋
args2[2] = (uint64_t)y_dsp;
args2[3] = (uint64_t)z;
args2[4] = (uint64_t)alpha_dsp;
args2[5] = (uint64_t)before2;
args2[6] = (uint64_t)after2;

int groupId = hthread_group_create(clusterId, nthreads);
if (groupId == -1) { fprintf(stderr, "group create failed\n"); return 2; }

uint64_t tD1=0, tD2=0, tC1=0, tC2=0;
uint64_t start, end;

// ---------- DSP执行 ----------
for (int k = 1; k < n; k++) {

    // kernel1 传当前迭代 k
    args1[0] = (uint64_t)k;
    start = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel1, 3, 7, args1); // 参数个数改8
    hthread_group_wait(groupId);
    end   = getCurrentTimeMicros();
    tD1 += end - start;

    // kernel2 同样传 k
    args2[0] = (uint64_t)k;
    start = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel2, 2, 5, args2);
    hthread_group_wait(groupId);
    end   = getCurrentTimeMicros();
    tD2 += end - start;
}

    // CPU 执行
    alpha_host = -r_host[0];
    beta_host = 1.0;
    y_host[0] = -r_host[0];
    for (int k = 1; k < n; k++) {
        start = getCurrentTimeMicros();
        durbin_cpu1(k, n, r_host, y_host, z_host, &alpha_host, &beta_host);
        end   = getCurrentTimeMicros();
        tC1 += end - start;

        start = getCurrentTimeMicros();
        durbin_cpu2(k, y_host, z_host, alpha_host);
        end   = getCurrentTimeMicros();
        tC2 += end - start;
    }

    // 校验
    int errNum = check_result(n, y_host, y_dsp);
    if (errNum == 0) {
        save_data(n, before1, after1, tD1, tC1, clusterId, devProgram, nthreads, kernel1);
        save_data(n, before2, after2, tD2, tC2, clusterId, devProgram, nthreads, kernel2);
        printf("WallTime Durbin_kernel1 (DSP/CPU): %fs / %fs\n", tD1 / 1e6, tC1 / 1e6);
        printf("WallTime Durbin_kernel2 (DSP/CPU): %fs / %fs\n", tD2 / 1e6, tC2 / 1e6);
    } else {
        fprintf(stderr, "Durbin test failed!\n");
    }

    // 清理
    hthread_free(r);
    hthread_free(y_dsp);
    hthread_free(z);
    hthread_free(alpha_dsp);
    hthread_free(beta_dsp);
    hthread_free(before1); hthread_free(after1);
    hthread_free(before2); hthread_free(after2);
    hthread_barrier_destroy(barrier_id);
    free(r_host);
    free(y_host);
    free(z_host);
    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return errNum ? 1 : 0;
}