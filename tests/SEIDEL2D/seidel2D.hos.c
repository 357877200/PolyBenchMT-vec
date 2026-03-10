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
void seidel_2d_cpu(int tsteps, int n, double *A)
{
    for (int t = 0; t < tsteps; t++) {
        for (int i = 1; i <= n - 2; i++) {
            for (int j = 1; j <= n - 2; j++) {
                A[i * n + j] = (A[(i - 1) * n + (j - 1)] + A[(i - 1) * n + j] + A[(i - 1) * n + (j + 1)]
                              + A[i * n + (j - 1)]     + A[i * n + j]     + A[i * n + (j + 1)]
                              + A[(i + 1) * n + (j - 1)] + A[(i + 1) * n + j] + A[(i + 1) * n + (j + 1)]) / 9.0;
            }
        }
    }
}

/*************************************
 * data initialisation / check
 ************************************/
void init_array(int n, double *A)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] = ((double)i * (j + 2) + 2) / n;
}

int check_result(int n, double *A_host, double *A_dev)
{
    int errNum = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (percentDiff(A_host[i * n + j], A_dev[i * n + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                if (errNum < 10) {
                    fprintf(stderr, "Diff @[%d][%d] : H=%.6f  D=%.6f\n",
                            i, j, A_host[i * n + j], A_dev[i * n + j]);
                }
                errNum++;
            }
        }
    }
    if (errNum) {
        double fail_percent = (100.0 * errNum) / (double)(n * n);
        fprintf(stderr,
                "Non-Matching CPU-DSP Beyond Error Threshold of %4.2f%% : %d (%.2lf%%)\n",
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
    FILE *fp = fopen("tests/SEIDEL2D/seidel_events.txt", "a");
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
    int    clusterId   = 1;
    int    n           = 256; // array dimension
    int    tsteps      = 8;   // time steps
    int    nthreads    = 1;
    char  *devProgram  = "operators/SEIDEL2D/seidel2D.dev.dat";
    char  *kernel1     = "seidel2d_kernel";

    // parse args
    for (int i = 1; i < argc; i++) {
        if (i + 1 >= argc) break;
        if      (!strcmp(argv[i], "--clusterId") || !strcmp(argv[i], "-c"))  { clusterId  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--n"))                                    { n         = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--tsteps"))                               { tsteps     = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--threads")  || !strcmp(argv[i], "-t"))   { nthreads   = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--program")  || !strcmp(argv[i], "-p"))   { devProgram = argv[++i]; }
        else if (!strcmp(argv[i], "--kernel1")  || !strcmp(argv[i], "-k1"))  { kernel1    = argv[++i]; }
    }

    if (access(devProgram, F_OK)) { fprintf(stderr, "%s not found\n", devProgram); return 2; }

    hthread_dev_open(clusterId);
    hthread_dat_load(clusterId, devProgram);

    size_t sizeMat = (size_t)n * n * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    double  *A_d     = (double *)hthread_malloc(clusterId, sizeMat, HT_MEM_RW);
    uint64_t *before1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after1  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);

    double *A_h = (double *)malloc(sizeMat);

    init_array(n, A_h);
    memcpy(A_d, A_h, sizeMat);
    memset(before1, 0, sizeHot);
    memset(after1, 0, sizeHot);


    // kernel args
    uint64_t args1[5];
    args1[0] = (uint64_t)tsteps;
    args1[1] = (uint64_t)n;
    args1[2] = (uint64_t)A_d;
    args1[3] = (uint64_t)before1;
    args1[4] = (uint64_t)after1;

    // DSP execute
    int groupId = hthread_group_create(clusterId, nthreads);
    uint64_t tDsp1 = 0, tCpu1 = 0;
    uint64_t st, ed;

    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel1, 2, 1, args1);
    hthread_group_wait(groupId);
    ed = getCurrentTimeMicros(); tDsp1 += ed - st;

    hthread_group_destroy(groupId);

    // CPU execute
    st = getCurrentTimeMicros();
    seidel_2d_cpu(tsteps, n, A_h);
    ed = getCurrentTimeMicros(); tCpu1 += ed - st;

    // check
    int err = check_result(n, A_h, A_d);
    if (err == 0) {
        save_data("SEIDEL2D", n, before1, after1, tDsp1, tCpu1,
                  clusterId, devProgram, nthreads, kernel1);
        printf("WallTime seidel2d_kernel (DSP/CPU): %fs / %fs\n",
               tDsp1 / 1e6, tCpu1 / 1e6);
    } else {
        fprintf(stderr, "seidel-2d test FAILED!\n");
    }

    // free
    hthread_free(A_d);
    hthread_free(before1);
    hthread_free(after1);
    free(A_h);

    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return err ? 1 : 0;
}