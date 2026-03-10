#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

#include "hthread_host.h"
#include "../common/tool.h" // percentDiff(), doubleToRawBits(), getCurrentTimeMicros()

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/*************************************
 * CPU reference kernel
 ************************************/
void cholesky_cpu(int n, double *A)
{
    for (int i = 0; i < n; i++) {
        // j < i case
        for (int j = 0; j < i; j++) {
            for (int k = 0; k < j; k++) {
                A[i * n + j] -= A[i * n + k] * A[j * n + k];
            }
            A[i * n + j] /= A[j * n + j];
        }
        // i == j case
        for (int k = 0; k < i; k++) {
            A[i * n + i] -= A[i * n + k] * A[i * n + k];
        }
        A[i * n + i] = sqrt(A[i * n + i]);
    }
}

/*************************************
 * data initialisation / check
 ************************************/
void init_array(int n, double *A)
{
    // 初始化下三角
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            A[i * n + j] = (double)(-j % n) / n + 1;
        }
        for (int j = i + 1; j < n; j++) {
            A[i * n + j] = 0.0;
        }
        A[i * n + i] = 1.0;
    }

    // 生成正半定矩阵：B = A * A^T
    double *B = (double *)malloc((size_t)n * n * sizeof(double));
    memset(B, 0, (size_t)n * n * sizeof(double));
    for (int t = 0; t < n; ++t)
        for (int r = 0; r < n; ++r)
            for (int s = 0; s < n; ++s)
                B[r * n + s] += A[r * n + t] * A[s * n + t];

    memcpy(A, B, (size_t)n * n * sizeof(double));
    free(B);
}

int check_result(int n, double *A_host, double *A_dev)
{
    int errNum = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) { // 只检查下三角结果
            if (percentDiff(A_host[i * n + j], A_dev[i * n + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                if (errNum < 10) {
                    fprintf(stderr, "A diff @[%d][%d] : H=%.6f  D=%.6f\n",
                            i, j, A_host[i * n + j], A_dev[i * n + j]);
                }
                errNum++;
            }
        }
    }
    if (errNum) {
        double fail_percent = (100.0 * errNum) / (double)(n * (n+1) / 2); // 下三角元素个数
        fprintf(stderr, "Non-Matching CPU-DSP Outputs Beyond Threshold: %d (%.2lf%%)\n",
                errNum, fail_percent);
    }
    return errNum;
}

/*************************************
 * save perf-counter data
 ************************************/
static void save_data(const char *bench,
                      int n,
                      uint64_t *before, uint64_t *after,
                      uint64_t tDsp, uint64_t tCpu,
                      int clusterId, const char *program,
                      int nthreads, const char *kernel)
{
    char path[256];
    snprintf(path, sizeof(path), "tests/CHOLESKY/cholesky_events.txt");
    FILE *fp = fopen(path, "a");
    if (!fp) { perror("fopen"); return; }
    fprintf(fp, "%s,%d,%d,%s,%s,%d,", bench, clusterId,
            n * n, program, kernel, nthreads);
    for (int i = 0; i < 26; i++) {
        fprintf(fp, "%lu", after[i] - before[i]);
        if (i != 25) fputc(',', fp);
    }
    fprintf(fp, ",%f,%f\n", tDsp / 1e6, tCpu / 1e6);
    fclose(fp);
}

/*************************************
 * main
 ************************************/
int main(int argc, char **argv)
{
    /* 默认参数 */
    int    clusterId   = 1;
    int    n           = 1000;
    int    nthreads    = 1;
    char  *devProgram  = "operators/CHOLESKY/cholesky.dev.dat";
    char  *kernel      = "cholesky_kernel";

    /* 解析命令行 */
    for (int i = 1; i < argc; i++) {
        if (i + 1 >= argc) break;
        if      (!strcmp(argv[i], "--clusterId") || !strcmp(argv[i], "-c"))  { clusterId  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--n"))                                    { n          = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--threads")  || !strcmp(argv[i], "-t"))   { nthreads   = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--program")  || !strcmp(argv[i], "-p"))   { devProgram = argv[++i]; }
        else if (!strcmp(argv[i], "--kernel")   || !strcmp(argv[i], "-k"))   { kernel     = argv[++i]; }
    }

    if (clusterId < 0 || clusterId > 3) { fprintf(stderr, "invalid clusterId\n"); return 2; }
    if (nthreads <= 0)                 { fprintf(stderr, "invalid nthreads\n");  return 2; }
    if (access(devProgram, F_OK))      { fprintf(stderr, "%s not found\n", devProgram); return 2; }

    /* 打开设备并加载程序 */
    int retc;
    retc = hthread_dev_open(clusterId);  if (retc) { fprintf(stderr, "dev open fail\n"); return retc; }
    retc = hthread_dat_load(clusterId, devProgram);
    if (retc) { fprintf(stderr, "load dat fail\n"); return retc; }

    int avail = hthread_get_avail_threads(clusterId);
    if (nthreads > avail) {
        fprintf(stderr, "thread overflow: avail %d, ask %d\n", avail, nthreads);
        hthread_dat_unload(clusterId); hthread_dev_close(clusterId); return 2;
    }

    /* 分配内存 */
    size_t sizeMatA = (size_t)n * n * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    double  *A_d  = (double *)hthread_malloc(clusterId, sizeMatA, HT_MEM_RW);
    uint64_t *before = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);

    if (!A_d || !before || !after) {
        fprintf(stderr, "device malloc failed\n"); return 1;
    }

    double *A_h = (double *)malloc(sizeMatA);
    if (!A_h) {
        fprintf(stderr, "malloc host fail\n"); return 1;
    }

    /* 初始化数据 */
    init_array(n, A_h);
    memcpy(A_d, A_h, sizeMatA);
    memset(before, 0, sizeHot); memset(after, 0, sizeHot);

    // 设备端同步单元
    int barrier_id = hthread_barrier_create(clusterId);
    
    /* kernel 参数 */
    uint64_t args[5];
    args[0] = (uint64_t)n;
    args[1] = (uint64_t)barrier_id;
    args[2] = (uint64_t)A_d;
    args[3] = (uint64_t)before;
    args[4] = (uint64_t)after;

    /* DSP 执行 */
    int groupId = hthread_group_create(clusterId, nthreads);
    if (groupId == -1) { fprintf(stderr, "group create fail\n"); return 2; }

    uint64_t tDsp = 0, tCpu = 0;
    uint64_t st, ed;

    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel, 2, 3, args);
    hthread_group_wait(groupId);
    ed = getCurrentTimeMicros(); tDsp = ed - st;

    hthread_group_destroy(groupId);

    /* CPU 执行 */
    st = getCurrentTimeMicros();
    cholesky_cpu(n, A_h);
    ed = getCurrentTimeMicros(); tCpu = ed - st;

    /* 校验结果 */
    int err = check_result(n, A_h, A_d);
    if (err != 0) {
        fprintf(stderr, "CHOLESKY test FAILED!\n");
    } else {
        save_data("CHOLESKY", n, before, after, tDsp, tCpu,
                  clusterId, devProgram, nthreads, kernel);
        printf("WallTime CHOLESKY_kernel (DSP/CPU): %fs / %fs\n",
               tDsp / 1e6, tCpu / 1e6);
    }

    /* 释放资源 */
    hthread_free(A_d);
    hthread_free(before);
    hthread_free(after);
    free(A_h);
    hthread_barrier_destroy(barrier_id);
    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return err ? 1 : 0;
}