#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>

#include "hthread_host.h"
#include "../common/tool.h"   // percentDiff, getCurrentTimeMicros

#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

/****************************************
 * CPU reference kernels
 ****************************************/
void heat3d_cpu1(int n, double *A, double *B)
{
    for (int i = 1; i < n-1; i++) {
        for (int j = 1; j < n-1; j++) {
            for (int k = 1; k < n-1; k++) {
                int idx = i*n*n + j*n + k;
                int ip = (i+1)*n*n + j*n + k;
                int im = (i-1)*n*n + j*n + k;
                int jp = i*n*n + (j+1)*n + k;
                int jm = i*n*n + (j-1)*n + k;
                int kp = i*n*n + j*n + (k+1);
                int km = i*n*n + j*n + (k-1);
                B[idx] = 0.125 * (A[ip] - 2.0*A[idx] + A[im])
                       + 0.125 * (A[jp] - 2.0*A[idx] + A[jm])
                       + 0.125 * (A[kp] - 2.0*A[idx] + A[km])
                       + A[idx];
            }
        }
    }
}

void heat3d_cpu2(int n, double *A, double *B)
{
    for (int i = 1; i < n-1; i++) {
        for (int j = 1; j < n-1; j++) {
            for (int k = 1; k < n-1; k++) {
                int idx = i*n*n + j*n + k;
                int ip = (i+1)*n*n + j*n + k;
                int im = (i-1)*n*n + j*n + k;
                int jp = i*n*n + (j+1)*n + k;
                int jm = i*n*n + (j-1)*n + k;
                int kp = i*n*n + j*n + (k+1);
                int km = i*n*n + j*n + (k-1);
                A[idx] = 0.125 * (B[ip] - 2.0*B[idx] + B[im])
                       + 0.125 * (B[jp] - 2.0*B[idx] + B[jm])
                       + 0.125 * (B[kp] - 2.0*B[idx] + B[km])
                       + B[idx];
            }
        }
    }
}

/****************************************
 * init / check
 ****************************************/
void init_array(int n, double *A, double *B)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++) {
                int idx = i*n*n + j*n + k;
                A[idx] = B[idx] = (double)(i + j + (n-k))*10.0 / n;
            }
}

int check_result(int n, double *A_host, double *A_dev)
{
    int errNum = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                int idx = i*n*n + j*n + k;
                if (percentDiff(A_host[idx], A_dev[idx]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                    if (errNum < 10) {
                        fprintf(stderr, "diff @[%d][%d][%d]: H=%.4f D=%.4f\n",
                                i, j, k, A_host[idx], A_dev[idx]);
                    }
                    errNum++;
                }
            }
        }
    }
    if (errNum) fprintf(stderr, "Total errors: %d\n", errNum);
    return errNum;
}
static void save_data(const char *bench,
                      int tmax, int nx, int ny,
                      uint64_t *before, uint64_t *after,
                      uint64_t tDsp, uint64_t tCpu,
                      int clusterId, const char *program,
                      int nthreads, const char *kernel)
{
    FILE *fp = fopen("tests/HEAT3D/heat3d_events.txt", "a");
    if (!fp) { perror("fopen"); return; }
    fprintf(fp, "%s,%d,%d,%s,%s,%d,", bench, clusterId,
            tmax*nx*ny, program, kernel, nthreads);
    for (int i = 0; i < 26; i++) {
        fprintf(fp, "%lu", after[i]-before[i]);
        if (i != 25) fputc(',', fp);
    }
    fprintf(fp, ",%f,%f\n", tDsp/1e6, tCpu/1e6);
    fclose(fp);
}
/****************************************
 * main
 ****************************************/
int main(int argc, char **argv)
{
    int clusterId = 1;
    int tsteps    = 20;
    int n         = 64;
    int nthreads  = 1;
    char *devProgram = "operators/HEAT3D/heat3d.dev.dat";
    char *kernel1    = "heat3d_kernel1";
    char *kernel2    = "heat3d_kernel2";

    // parse cmdline
    for (int i = 1; i < argc; i++) {
        if (i+1 >= argc) break;
        if      (!strcmp(argv[i], "--clusterId") || !strcmp(argv[i], "-c")) { clusterId  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--tsteps"))                              { tsteps     = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--n"))                                   { n          = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--threads") || !strcmp(argv[i], "-t"))   { nthreads   = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--program") || !strcmp(argv[i], "-p"))   { devProgram = argv[++i]; }
        else if (!strcmp(argv[i], "--kernel1") || !strcmp(argv[i], "-k1"))  { kernel1    = argv[++i]; }
        else if (!strcmp(argv[i], "--kernel2") || !strcmp(argv[i], "-k2"))  { kernel2    = argv[++i]; }
    }

    if (clusterId < 0 || clusterId > 3) { fprintf(stderr, "invalid clusterId\n"); return 2; }
    if (nthreads <= 0)                  { fprintf(stderr, "invalid nthreads\n"); return 2; }
    if (access(devProgram, F_OK))       { fprintf(stderr, "%s not found\n", devProgram); return 2; }

    // open device
    int retc;
    retc = hthread_dev_open(clusterId);
    if (retc) { fprintf(stderr, "dev open fail\n"); return retc; }
    retc = hthread_dat_load(clusterId, devProgram);
    if (retc) { fprintf(stderr, "load dat fail\n"); return retc; }

    int avail = hthread_get_avail_threads(clusterId);
    if (nthreads > avail) {
        fprintf(stderr, "thread overflow: avail %d, ask %d\n", avail, nthreads);
        hthread_dat_unload(clusterId);
        hthread_dev_close(clusterId);
        return 2;
    }

    // alloc memory
    size_t sizeA   = (size_t)n*n*n * sizeof(double);
    size_t sizeB   = (size_t)n*n*n * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    // device
    double *A_d = (double *)hthread_malloc(clusterId, sizeA, HT_MEM_RW);
    double *B_d = (double *)hthread_malloc(clusterId, sizeB, HT_MEM_RW);
    uint64_t *before1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after1  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after2  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);

    if (!A_d || !B_d || !before1 || !after1 || !before2 || !after2) {
        fprintf(stderr, "device malloc failed\n");
        return 1;
    }

    // host
    double *A_h = (double *)malloc(sizeA);
    double *B_h = (double *)malloc(sizeB);
    if (!A_h || !B_h) { fprintf(stderr, "malloc host fail\n"); return 1; }

    // init data
    init_array(n, A_h, B_h);
    memcpy(A_d, A_h, sizeA);
    memcpy(B_d, B_h, sizeB);
    memset(before1, 0, sizeHot); memset(after1, 0, sizeHot);
    memset(before2, 0, sizeHot); memset(after2, 0, sizeHot);

    // kernel args
    uint64_t args1[5];
    args1[0] = (uint64_t)n;
    args1[1] = (uint64_t)A_d;
    args1[2] = (uint64_t)B_d;
    args1[3] = (uint64_t)before1;
    args1[4] = (uint64_t)after1;

    uint64_t args2[5];
    args2[0] = (uint64_t)n;
    args2[1] = (uint64_t)A_d;
    args2[2] = (uint64_t)B_d;
    args2[3] = (uint64_t)before2;
    args2[4] = (uint64_t)after2;

    // DSP exec
    int groupId = hthread_group_create(clusterId, nthreads);
    if (groupId == -1) { fprintf(stderr, "group create fail\n"); return 2; }

    uint64_t tDsp1=0, tDsp2=0, tCpu1=0, tCpu2=0;
    uint64_t st, ed;
    for (int t = 0; t < tsteps; t++) {
        st = getCurrentTimeMicros();
        hthread_group_exec(groupId, kernel1, 1, 4, args1);
        hthread_group_wait(groupId);
        ed = getCurrentTimeMicros(); tDsp1 += ed - st;

        st = getCurrentTimeMicros();
        hthread_group_exec(groupId, kernel2, 1, 4, args2);
        hthread_group_wait(groupId);
        ed = getCurrentTimeMicros(); tDsp2 += ed - st;
    }
    hthread_group_destroy(groupId);

    // CPU exec
    for (int t = 0; t < tsteps; t++) {
        st = getCurrentTimeMicros();
        heat3d_cpu1(n, A_h, B_h);
        ed = getCurrentTimeMicros(); tCpu1 += ed - st;

        st = getCurrentTimeMicros();
        heat3d_cpu2(n, A_h, B_h);
        ed = getCurrentTimeMicros(); tCpu2 += ed - st;
    }

    // check
    int err = check_result(n, A_h, A_d);
    if (err != 0) {
        fprintf(stderr, "HEAT3D test FAILED!\n");
    } else {
        save_data("HEAT3D", tsteps, n, n, before1, after1, tDsp1, tCpu1,
                  clusterId, devProgram, nthreads, kernel1);
        save_data("HEAT3D", tsteps, n, n, before2, after2, tDsp2, tCpu2,
                  clusterId, devProgram, nthreads, kernel2);
        printf("WallTime HEAT3D_kernel1 (DSP/CPU): %fs / %fs\n", tDsp1/1e6, tCpu1/1e6);
        printf("WallTime HEAT3D_kernel2 (DSP/CPU): %fs / %fs\n", tDsp2/1e6, tCpu2/1e6);
    }

    // free
    hthread_free(A_d); hthread_free(B_d);
    hthread_free(before1); hthread_free(after1);
    hthread_free(before2); hthread_free(after2);
    free(A_h); free(B_h);

    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);
    return err ? 1 : 0;
}