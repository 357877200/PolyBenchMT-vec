#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

#include "hthread_host.h"
#include "../common/tool.h" // percentDiff(), getCurrentTimeMicros()

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/*************************************
 * CPU reference kernel
 ************************************/
void trisolv_cpu(int n, double *L, double *x, double *b)
{
    for (int i = 0; i < n; i++) {
        x[i] = b[i];
        for (int j = 0; j < i; j++)
            x[i] -= L[i * n + j] * x[j];
        x[i] = x[i] / L[i * n + i];
    }
}

/*************************************
 * data initialisation / check
 ************************************/
void init_array(int n, double *L, double *x, double *b)
{
    for (int i = 0; i < n; i++) {
        x[i] = -999;
        b[i] = i;
        for (int j = 0; j <= i; j++)
            L[i * n + j] = ((double)(i + n - j + 1) * 2) / n;
        for (int j = i + 1; j < n; j++)
            L[i * n + j] = 0.0; // 上三角无效区置零
    }
}

int check_result(int n, double *x_host, double *x_dev)
{
    int errNum = 0;
    for (int i = 0; i < n; i++) {
        if (percentDiff(x_host[i], x_dev[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
            if (errNum < 10) {
                fprintf(stderr, "x diff @[%d] : H=%.6f  D=%.6f\n",
                        i, x_host[i], x_dev[i]);
            }
            errNum++;
        }
    }
    if (errNum) {
        double fail_percent = (100.0 * errNum) / (double)n;
        fprintf(stderr,
                "Non-Matching CPU-DSP Outputs Beyond Error Threshold of %4.2f%% : %d (%.2lf%%)\n",
                PERCENT_DIFF_ERROR_THRESHOLD, errNum, fail_percent);
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
    FILE *fp = fopen("tests/TRISOLV/trisolv_events.txt", "a");
    if (!fp) { perror("fopen"); return; }
    fprintf(fp, "%s,%d,%d,%s,%s,%d,", bench, clusterId,
            n, program, kernel, nthreads);
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
    int    clusterId   = 1;
    int    n           = 1024; // problem size
    int    nthreads    = 1;
    char  *devProgram  = "operators/TRISOLV/trisolv.dev.dat";
    char  *kernel1     = "trisolv_kernel";

    // parse args
    for (int i = 1; i < argc; i++) {
        if (i + 1 >= argc) break;
        if      (!strcmp(argv[i], "--clusterId") || !strcmp(argv[i], "-c"))  { clusterId  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--n"))                                    { n         = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--threads")  || !strcmp(argv[i], "-t"))   { nthreads   = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--program")  || !strcmp(argv[i], "-p"))   { devProgram = argv[++i]; }
        else if (!strcmp(argv[i], "--kernel1")  || !strcmp(argv[i], "-k1"))  { kernel1    = argv[++i]; }
    }

    if (access(devProgram, F_OK)) { fprintf(stderr, "%s not found\n", devProgram); return 2; }

    hthread_dev_open(clusterId);
    hthread_dat_load(clusterId, devProgram);

    size_t sizeMat = (size_t)n * n * sizeof(double);
    size_t sizeVec = (size_t)n * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    double  *L_d     = (double *)hthread_malloc(clusterId, sizeMat, HT_MEM_RW);
    double  *x_d     = (double *)hthread_malloc(clusterId, sizeVec, HT_MEM_RW);
    double  *b_d     = (double *)hthread_malloc(clusterId, sizeVec, HT_MEM_RW);
    uint64_t *before1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after1  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);

    double *L_h = (double *)malloc(sizeMat);
    double *x_h = (double *)malloc(sizeVec);
    double *b_h = (double *)malloc(sizeVec);

    init_array(n, L_h, x_h, b_h);
    memcpy(L_d, L_h, sizeMat);
    memcpy(x_d, x_h, sizeVec);
    memcpy(b_d, b_h, sizeVec);
    memset(before1, 0, sizeHot);
    memset(after1, 0, sizeHot);

    // kernel args
    uint64_t args1[6];
    args1[0] = (uint64_t)n;
    args1[1] = (uint64_t)L_d;
    args1[2] = (uint64_t)x_d;
    args1[3] = (uint64_t)b_d;
    args1[4] = (uint64_t)before1;
    args1[5] = (uint64_t)after1;

    // DSP execute
    int groupId = hthread_group_create(clusterId, nthreads);
    uint64_t tDsp1 = 0, tCpu1 = 0;
    uint64_t st, ed;

    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel1, 1,3, args1);
    hthread_group_wait(groupId);
    ed = getCurrentTimeMicros(); tDsp1 += ed - st;

    hthread_group_destroy(groupId);

    // CPU execute
    st = getCurrentTimeMicros();
    trisolv_cpu(n, L_h, x_h, b_h);
    ed = getCurrentTimeMicros(); tCpu1 += ed - st;

    // check
    int err = check_result(n, x_h, x_d);
    if (err == 0) {
        save_data("TRISOLV", n, before1, after1, tDsp1, tCpu1,
                  clusterId, devProgram, nthreads, kernel1);
        printf("WallTime trisolv_kernel (DSP/CPU): %fs / %fs\n",
               tDsp1 / 1e6, tCpu1 / 1e6);
    } else {
        fprintf(stderr, "trisolv test FAILED!\n");
    }

    // free
    hthread_free(L_d);
    hthread_free(x_d);
    hthread_free(b_d);
    hthread_free(before1);
    hthread_free(after1);
    free(L_h);
    free(x_h);
    free(b_h);

    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return err ? 1 : 0;
}