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
 * CPU reference kernel
 ************************************/
void trmm_cpu(int m, int n, double alpha, double *A, double *B)
{
    // BLAS 参数: SIDE='L', UPLO='L', TRANSA='T', DIAG='U'
    // 公式: B := alpha * A^T * B
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = i + 1; k < m; k++) {
                B[i * n + j] += A[k * m + i] * B[k * n + j];
            }
            B[i * n + j] = alpha * B[i * n + j];
        }
    }
}

/*************************************
 * data init / check
 ************************************/
void init_arrays(int m, int n, double *alpha, double *A, double *B)
{
    *alpha = 1.5;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            if (j < i)
                A[i * m + j] = ((double)((i + j) % m)) / m;
            else if (i == j)
                A[i * m + j] = 1.0;
            else
                A[i * m + j] = 0.0; // 上三角补零
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            B[i * n + j] = ((double)((n + (i - j)) % n)) / n;
        }
    }
}

int check_result(int m, int n, double *B_host, double *B_dev)
{
    int errNum = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (percentDiff(B_host[i * n + j], B_dev[i * n + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                if (errNum < 10) {
                    fprintf(stderr, "B diff @[%d][%d] : H=%.6f  D=%.6f\n",
                            i, j, B_host[i * n + j], B_dev[i * n + j]);
                }
                errNum++;
            }
        }
    }
    if (errNum) {
        double fail_percent = (100.0 * errNum) / (double)(m * n);
        fprintf(stderr, "Non-Matching CPU-DSP Outputs Beyond Error Threshold of %4.2f Percent: %d (%.2lf%%)\n",
                PERCENT_DIFF_ERROR_THRESHOLD, errNum, fail_percent);
    }
    return errNum;
}

static void save_data(const char *bench,
                      int size,
                      uint64_t *before, uint64_t *after,
                      uint64_t tDsp, uint64_t tCpu,
                      int clusterId, const char *program,
                      int nthreads, const char *kernel)
{
    FILE *fp = fopen("tests/TRMM/trmm_events.txt", "a");
    if (!fp) { perror("fopen"); return; }
    fprintf(fp, "%s,%d,%d,%s,%s,%d,", bench, clusterId,
            size, program, kernel, nthreads);
    for (int i = 0; i < 26; i++) {
        fprintf(fp, "%lu", after[i] - before[i]);
        if (i != 25) fputc(',', fp);
    }
    fprintf(fp, ",%f,%f\n", tDsp / 1e6, tCpu / 1e6);
    fclose(fp);
}

int main(int argc, char **argv)
{
    int clusterId   = 1;
    int m           = 1000; // M
    int n           = 1000; // N
    int nthreads    = 1;
    char *devProgram  = "operators/TRMM/trmm.dev.dat";
    char *kernel      = "trmm_kernel";

    for (int i = 1; i < argc; i++) {
        if (i + 1 >= argc) break;
        if      (!strcmp(argv[i], "--clusterId") || !strcmp(argv[i], "-c")) { clusterId  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--m"))                                   { m         = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--n"))                                   { n         = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--threads")  || !strcmp(argv[i], "-t"))  { nthreads  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--program")  || !strcmp(argv[i], "-p"))  { devProgram= argv[++i]; }
        else if (!strcmp(argv[i], "--kernel")   || !strcmp(argv[i], "-k"))  { kernel    = argv[++i]; }
    }

    if (clusterId < 0 || clusterId > 3) { fprintf(stderr, "invalid clusterId\n"); return 2; }
    if (nthreads <= 0)                 { fprintf(stderr, "invalid nthreads\n");  return 2; }
    if (access(devProgram, F_OK))      { fprintf(stderr, "%s not found\n", devProgram); return 2; }

    int retc;
    retc = hthread_dev_open(clusterId);  if (retc) { fprintf(stderr, "dev open fail\n"); return retc; }
    retc = hthread_dat_load(clusterId, devProgram);
    if (retc) { fprintf(stderr, "load dat fail\n"); return retc; }

    int avail = hthread_get_avail_threads(clusterId);
    if (nthreads > avail) {
        fprintf(stderr, "thread overflow: avail %d, ask %d\n", avail, nthreads);
        hthread_dat_unload(clusterId); hthread_dev_close(clusterId); return 2;
    }

    size_t sizeMatA = (size_t)m * m * sizeof(double);
    size_t sizeMatB = (size_t)m * n * sizeof(double);
    size_t sizeHot  = 26 * sizeof(uint64_t);

    double  *A_d  = (double *)hthread_malloc(clusterId, sizeMatA, HT_MEM_RO);
    double  *B_d  = (double *)hthread_malloc(clusterId, sizeMatB, HT_MEM_RW);
    uint64_t *before = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    if (!A_d || !B_d || !before || !after) { fprintf(stderr, "device malloc failed\n"); return 1; }

    double *A_h = (double *)malloc(sizeMatA);
    double *B_h = (double *)malloc(sizeMatB);
    if (!A_h || !B_h) { fprintf(stderr, "malloc host fail\n"); return 1; }

    double alpha;
    init_arrays(m, n, &alpha, A_h, B_h);

    memcpy(A_d, A_h, sizeMatA);
    memcpy(B_d, B_h, sizeMatB);
    memset(before, 0, sizeHot);
    memset(after, 0, sizeHot);

    uint64_t args[7];
    args[0] = (uint64_t)m;
    args[1] = (uint64_t)n;
    args[2] = doubleToRawBits(alpha);
    args[3] = (uint64_t)A_d;
    args[4] = (uint64_t)B_d;
    args[5] = (uint64_t)before;
    args[6] = (uint64_t)after;

    int groupId = hthread_group_create(clusterId, nthreads);
    if (groupId == -1) { fprintf(stderr, "group create fail\n"); return 2; }

    uint64_t tDsp = 0, tCpu = 0;
    uint64_t st, ed;

    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel, 4, 3, args);
    hthread_group_wait(groupId);
    ed = getCurrentTimeMicros(); tDsp = ed - st;

    hthread_group_destroy(groupId);

    st = getCurrentTimeMicros();
    trmm_cpu(m, n, alpha, A_h, B_h);
    ed = getCurrentTimeMicros(); tCpu = ed - st;

    int err = check_result(m, n, B_h, B_d);
    if (err != 0) {
        fprintf(stderr, "TRMM test FAILED!\n");
    } else {
        save_data("TRMM", m * n, before, after, tDsp, tCpu,
                  clusterId, devProgram, nthreads, kernel);
        printf("WallTime TRMM_kernel (DSP/CPU): %fs / %fs\n",tDsp / 1e6, tCpu / 1e6);
    }

    hthread_free(A_d);
    hthread_free(B_d);
    hthread_free(before);
    hthread_free(after);

    free(A_h);
    free(B_h);

    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return err ? 1 : 0;
}