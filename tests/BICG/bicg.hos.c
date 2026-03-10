#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

#include "hthread_host.h"
#include "../common/tool.h"          // percentDiff()、getCurrentTimeMicros()等

#define PERCENT_DIFF_ERROR_THRESHOLD 0.5
#ifndef M_PI
#define M_PI 3.14159
#endif

/*************************************
 * CPU reference kernels
 ************************************/
void bicg_cpu1(int nx, int ny, double *A, double *r, double *s)
{
    for (int j = 0; j < ny; j++) s[j] = 0.0;

    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            s[j] += r[i] * A[i * ny + j];
}

void bicg_cpu2(int nx, int ny, double *A, double *p, double *q)
{
    for (int i = 0; i < nx; i++) q[i] = 0.0;

    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            q[i] += A[i * ny + j] * p[j];
}

/*************************************
 * data initialisation / check
 ************************************/
void init_array(int nx, int ny, double *A, double *p, double *r)
{
    for (int i = 0; i < ny; i++)          p[i] = i * M_PI;
    for (int i = 0; i < nx; i++) {
        r[i] = i * M_PI;
        for (int j = 0; j < ny; j++)
            A[i * ny + j] = ((double)i * j) / nx;
    }
}

int check_result(int nx, int ny,
                 double *s_host, double *q_host,
                 double *s_dev , double *q_dev)
{
    int errNum = 0;
    for (int i = 0; i < ny; i++)
        if (percentDiff(s_host[i], s_dev[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
            if (errNum < 10)
                fprintf(stderr, "s diff @%d : H=%.4f  D=%.4f\n",
                        i, s_host[i], s_dev[i]);
            errNum++;
        }

    for (int i = 0; i < nx; i++)
        if (percentDiff(q_host[i], q_dev[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
            if (errNum < 10)
                fprintf(stderr, "q diff @%d : H=%.4f  D=%.4f\n",
                        i, q_host[i], q_dev[i]);
            errNum++;
        }
    if (errNum)  fprintf(stderr, "Total errors : %d\n", errNum);
    return errNum;
}

/*************************************
 * save perf-counter data
 ************************************/
static void save_data(const char *bench,
                      int nx, int ny,
                      uint64_t *before, uint64_t *after,
                      uint64_t tDsp, uint64_t tCpu,
                      int clusterId, const char *program,
                      int nthreads, const char *kernel)
{
    FILE *fp = fopen("tests/BICG/bicg_events.txt", "a");
    if (!fp) { perror("fopen"); return; }
    fprintf(fp, "%s,%d,%d,%s,%s,%d,", bench, clusterId,
            nx*ny, program, kernel, nthreads);
    for (int i = 0; i < 26; i++) {
        fprintf(fp, "%lu", after[i]-before[i]);
        if (i != 25) fputc(',', fp);
    }
    fprintf(fp, ",%f,%f\n", tDsp/1e6, tCpu/1e6);
    fclose(fp);
}

/*************************************
 * main
 ************************************/
int main(int argc, char **argv)
{
    /* -------------------- 默认参数 -------------------- */
    int    clusterId   = 1;
    int    nx          = 128;
    int    ny          = 128;
    int    nthreads    = 1;
    char  *devProgram  = "operators/BICG/bicg.dev.dat";
    char  *kernel1     = "bicg_kernel1";
    char  *kernel2     = "bicg_kernel2";

    /* -------------------- 解析命令行 ------------------ */
    for (int i = 1; i < argc; i++) {
        if (i+1 >= argc) break;
        if      (!strcmp(argv[i], "--clusterId") || !strcmp(argv[i], "-c"))  { clusterId  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--nx"))                                   { nx         = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--ny"))                                   { ny         = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--threads")  || !strcmp(argv[i], "-t"))   { nthreads   = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--program")  || !strcmp(argv[i], "-p"))   { devProgram = argv[++i]; }
        else if (!strcmp(argv[i], "--kernel1")  || !strcmp(argv[i], "-k1"))  { kernel1    = argv[++i]; }
        else if (!strcmp(argv[i], "--kernel2")  || !strcmp(argv[i], "-k2"))  { kernel2    = argv[++i]; }
    }

    /* -------------------- 参数合法性 ------------------ */
    if (clusterId < 0 || clusterId > 3) { fprintf(stderr, "invalid clusterId\n"); return 2; }
    if (nthreads <= 0)                 { fprintf(stderr, "invalid nthreads\n");  return 2; }
    if (access(devProgram, F_OK))      { fprintf(stderr, "%s not found\n", devProgram); return 2; }

    /* -------------------- 打开设备 -------------------- */
    int retc;
    retc = hthread_dev_open(clusterId);  if (retc) { fprintf(stderr, "dev open fail\n"); return retc; }
    retc = hthread_dat_load(clusterId, devProgram);
    if (retc) { fprintf(stderr, "load dat fail\n"); return retc; }

    int avail = hthread_get_avail_threads(clusterId);
    if (nthreads > avail) {
        fprintf(stderr, "thread overflow: avail %d, ask %d\n", avail, nthreads);
        hthread_dat_unload(clusterId); hthread_dev_close(clusterId); return 2;
    }

    /* -------------------- 内存分配 -------------------- */
    size_t sizeA = (size_t)nx * ny * sizeof(double);
    size_t sizeX = (size_t)nx * sizeof(double);
    size_t sizeY = (size_t)ny * sizeof(double);
    size_t sizeHot= 26 * sizeof(uint64_t);

    /* device */
    double  *A   = (double *)hthread_malloc(clusterId, sizeA, HT_MEM_RO);
    double  *p   = (double *)hthread_malloc(clusterId, sizeY, HT_MEM_RO);
    double  *r   = (double *)hthread_malloc(clusterId, sizeX, HT_MEM_RO);
    double  *s_d = (double *)hthread_malloc(clusterId, sizeY, HT_MEM_RW);
    double  *q_d = (double *)hthread_malloc(clusterId, sizeX, HT_MEM_RW);

    uint64_t *before1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after1  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after2  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);

    if (!A || !p || !r || !s_d || !q_d || !before1 || !after1 || !before2 || !after2) {
        fprintf(stderr, "device malloc failed\n"); return 1;
    }

    /* host */
    double *A_h = (double *)malloc(sizeA);
    double *p_h = (double *)malloc(sizeY);
    double *r_h = (double *)malloc(sizeX);
    double *s_h = (double *)malloc(sizeY);
    double *q_h = (double *)malloc(sizeX);
    if (!A_h||!p_h||!r_h||!s_h||!q_h){fprintf(stderr,"malloc host fail\n");return 1;}

    /* init data */
    init_array(nx, ny, A_h, p_h, r_h);
    memcpy(A , A_h, sizeA);
    memcpy(p , p_h, sizeY);
    memcpy(r , r_h, sizeX);
    memset(s_d, 0, sizeY);
    memset(q_d, 0, sizeX);
    memset(s_h, 0, sizeY);
    memset(q_h, 0, sizeX);
    memset(before1, 0, sizeHot); memset(after1, 0, sizeHot);
    memset(before2, 0, sizeHot); memset(after2, 0, sizeHot);

    /* -------------------- kernel 参数 ----------------- */
    uint64_t args1[7];
    args1[0] = (uint64_t)nx;
    args1[1] = (uint64_t)ny;
    args1[2] = (uint64_t)A;
    args1[3] = (uint64_t)r;
    args1[4] = (uint64_t)s_d;
    args1[5] = (uint64_t)before1;
    args1[6] = (uint64_t)after1;

    uint64_t args2[7];
    args2[0] = (uint64_t)nx;
    args2[1] = (uint64_t)ny;
    args2[2] = (uint64_t)A;
    args2[3] = (uint64_t)p;
    args2[4] = (uint64_t)q_d;
    args2[5] = (uint64_t)before2;
    args2[6] = (uint64_t)after2;

    /* -------------------- DSP 执行 -------------------- */
    int groupId = hthread_group_create(clusterId, nthreads);
    if (groupId == -1) { fprintf(stderr, "group create fail\n"); return 2; }

    uint64_t tDsp1=0,tDsp2=0,tCpu1=0,tCpu2=0;
    uint64_t st,ed;

    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel1, 2, 5, args1);
    hthread_group_wait(groupId);
    ed = getCurrentTimeMicros();  tDsp1 = ed - st;

    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel2, 2, 5, args2);
    hthread_group_wait(groupId);
    ed = getCurrentTimeMicros();  tDsp2 = ed - st;

    hthread_group_destroy(groupId);

    /* -------------------- CPU 执行 -------------------- */
    st = getCurrentTimeMicros();
    bicg_cpu1(nx, ny, A_h, r_h, s_h);
    ed = getCurrentTimeMicros();  tCpu1 = ed - st;

    st = getCurrentTimeMicros();
    bicg_cpu2(nx, ny, A_h, p_h, q_h);
    ed = getCurrentTimeMicros();  tCpu2 = ed - st;

    /* -------------------- 校验结果 ------------------- */
    int err = check_result(nx, ny, s_h, q_h, s_d, q_d);
    if (err != 0) {
        fprintf(stderr, "BICG test FAILED!\n");
    } else {
        save_data("BICG", nx, ny, before1, after1, tDsp1, tCpu1,
                  clusterId, devProgram, nthreads, kernel1);
        save_data("BICG", nx, ny, before2, after2, tDsp2, tCpu2,
                  clusterId, devProgram, nthreads, kernel2);
        printf("WallTime BICG_kernel1 (DSP/CPU): %fs / %fs\n",
                tDsp1/1e6, tCpu1/1e6);
        printf("WallTime BICG_kernel2 (DSP/CPU): %fs / %fs\n",
                tDsp2/1e6, tCpu2/1e6);
    }

    /* -------------------- 资源释放 -------------------- */
    hthread_free(A);  hthread_free(p);  hthread_free(r);
    hthread_free(s_d);hthread_free(q_d);
    hthread_free(before1); hthread_free(after1);
    hthread_free(before2); hthread_free(after2);

    free(A_h); free(p_h); free(r_h); free(s_h); free(q_h);

    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return err ? 1 : 0;
}