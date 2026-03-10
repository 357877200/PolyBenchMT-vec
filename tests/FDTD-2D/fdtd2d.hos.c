#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

#include "hthread_host.h"
#include "../common/tool.h"          // percentDiff()、getCurrentTimeMicros()等

#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

/*************************************
 * CPU reference kernels
 ************************************/
void fdtd_cpu1(int t, int nx, int ny, double *_fict_, double *ex, double *ey, double *hz)
{
    // kernel1: 边界条件设置
    for (int j = 0; j < ny; j++) {
        ey[0 * ny + j] = _fict_[t];
    }
}

void fdtd_cpu2(int t, int nx, int ny, double *_fict_, double *ex, double *ey, double *hz)
{
    // kernel2: 更新ey
    for (int i = 1; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ey[i * ny + j] = ey[i * ny + j] - 0.5 * (hz[i * ny + j] - hz[(i - 1) * ny + j]);
        }
    }
}
void fdtd_cpu3(int t, int nx, int ny, double *_fict_, double *ex, double *ey, double *hz)
{
    // kernel3: 更新ex和hz
    for (int i = 0; i < nx; i++) {
        for (int j = 1; j < ny; j++) {
            ex[i * ny + j] = ex[i * ny + j] - 0.5 * (hz[i * ny + j] - hz[i * ny + (j - 1)]);
        }
    }
    
    for (int i = 0; i < nx - 1; i++) {
        for (int j = 0; j < ny - 1; j++) {
            hz[i * ny + j] = hz[i * ny + j] - 0.7 * (ex[i * ny + (j + 1)] - ex[i * ny + j] + ey[(i + 1) * ny + j] - ey[i * ny + j]);
        }
    }
}

/*************************************
 * data initialisation / check
 ************************************/
void init_array(int tmax, int nx, int ny, double *_fict_, double *ex, double *ey, double *hz)
{
    for (int i = 0; i < tmax; i++) {
        _fict_[i] = (double)i;
    }

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ex[i * ny + j] = ((double)i * (j + 1) + 1) / nx;
            ey[i * ny + j] = ((double)(i - 1) * (j + 2) + 2) / nx;
            hz[i * ny + j] = ((double)(i - 9) * (j + 4) + 3) / nx;
        }
    }
}

int check_result(int nx, int ny, double *hz_host, double *hz_dev)
{
    int errNum = 0;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            int idx = i * ny + j;
            if (percentDiff(hz_host[idx], hz_dev[idx]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                if (errNum < 10) {
                    fprintf(stderr, "hz diff @[%d][%d] : H=%.4f  D=%.4f\n",
                            i, j, hz_host[idx], hz_dev[idx]);
                }
                errNum++;
            }
        }
    }
    if (errNum) fprintf(stderr, "Total errors : %d\n", errNum);
    return errNum;
}

/*************************************
 * save perf-counter data
 ************************************/
static void save_data(const char *bench,
                      int tmax, int nx, int ny,
                      uint64_t *before, uint64_t *after,
                      uint64_t tDsp, uint64_t tCpu,
                      int clusterId, const char *program,
                      int nthreads, const char *kernel)
{
    FILE *fp = fopen("tests/FDTD-2D/fdtd2d_events.txt", "a");
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

/*************************************
 * main
 ************************************/
int main(int argc, char **argv)
{
    /* -------------------- 默认参数 -------------------- */
    int    clusterId   = 1;
    int    tmax        = 20;
    int    nx          = 64;
    int    ny          = 64;
    int    nthreads    = 1;
    char  *devProgram  = "operators/FDTD-2D/fdtd2d.dev.dat";
    char  *kernel1     = "fdtd2d_kernel1";
    char  *kernel2     = "fdtd2d_kernel2";
    char  *kernel3     = "fdtd2d_kernel3";

    /* -------------------- 解析命令行 ------------------ */
    for (int i = 1; i < argc; i++) {
        if (i+1 >= argc) break;
        if      (!strcmp(argv[i], "--clusterId") || !strcmp(argv[i], "-c"))  { clusterId  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--tmax"))                                 { tmax       = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--nx"))                                   { nx         = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--ny"))                                   { ny         = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--threads")  || !strcmp(argv[i], "-t"))   { nthreads   = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--program")  || !strcmp(argv[i], "-p"))   { devProgram = argv[++i]; }
        else if (!strcmp(argv[i], "--kernel1")  || !strcmp(argv[i], "-k1"))  { kernel1    = argv[++i]; }
        else if (!strcmp(argv[i], "--kernel2")  || !strcmp(argv[i], "-k2"))  { kernel2    = argv[++i]; }
        else if (!strcmp(argv[i], "--kernel3")  || !strcmp(argv[i], "-k3"))  { kernel3    = argv[++i]; }
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
    size_t sizeFict = (size_t)tmax * sizeof(double);
    size_t sizeEx   = (size_t)nx * ny * sizeof(double);
    size_t sizeEy   = (size_t)nx * ny * sizeof(double);
    size_t sizeHz   = (size_t)nx * ny * sizeof(double);
    size_t sizeHot  = 26 * sizeof(uint64_t);

    /* device */
    double  *_fict_d = (double *)hthread_malloc(clusterId, sizeFict, HT_MEM_RO);
    double  *ex_d    = (double *)hthread_malloc(clusterId, sizeEx, HT_MEM_RW);
    double  *ey_d    = (double *)hthread_malloc(clusterId, sizeEy, HT_MEM_RW);
    double  *hz_d    = (double *)hthread_malloc(clusterId, sizeHz, HT_MEM_RW);

    uint64_t *before1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after1  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after2  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before3 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after3  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);

    if (!_fict_d || !ex_d || !ey_d || !hz_d || !before1 || !after1 || !before2 || !after2 || !before3 || !after3) {
        fprintf(stderr, "device malloc failed\n"); return 1;
    }

    /* host */
    double *_fict_h = (double *)malloc(sizeFict);
    double *ex_h    = (double *)malloc(sizeEx);
    double *ey_h    = (double *)malloc(sizeEy);
    double *hz_h    = (double *)malloc(sizeHz);
    if (!_fict_h || !ex_h || !ey_h || !hz_h) { fprintf(stderr, "malloc host fail\n"); return 1; }

    /* init data */
    init_array(tmax, nx, ny, _fict_h, ex_h, ey_h, hz_h);
    memcpy(_fict_d, _fict_h, sizeFict);
    memcpy(ex_d, ex_h, sizeEx);
    memcpy(ey_d, ey_h, sizeEy);
    memcpy(hz_d, hz_h, sizeHz);
    memset(before1, 0, sizeHot); memset(after1, 0, sizeHot);
    memset(before2, 0, sizeHot); memset(after2, 0, sizeHot);
    memset(before3, 0, sizeHot); memset(after3, 0, sizeHot);

    /* -------------------- kernel 参数 ----------------- */
    uint64_t args1[9];
    args1[0] = (uint64_t)nx;
    args1[1] = (uint64_t)ny;
    args1[3] = (uint64_t)_fict_d;
    args1[4] = (uint64_t)ex_d;
    args1[5] = (uint64_t)ey_d;
    args1[6] = (uint64_t)hz_d;
    args1[7] = (uint64_t)before1;
    args1[8] = (uint64_t)after1;

    uint64_t args2[8];
    args2[0] = (uint64_t)nx;
    args2[1] = (uint64_t)ny;
    args2[3] = (uint64_t)ex_d;
    args2[4] = (uint64_t)ey_d;
    args2[5] = (uint64_t)hz_d;
    args2[6] = (uint64_t)before2;
    args2[7] = (uint64_t)after2;

    uint64_t args3[8];
    args3[0] = (uint64_t)nx;
    args3[1] = (uint64_t)ny;
    args3[3] = (uint64_t)ex_d;
    args3[4] = (uint64_t)ey_d;
    args3[5] = (uint64_t)hz_d;
    args3[6] = (uint64_t)before3;
    args2[7] = (uint64_t)after2;

    /* -------------------- DSP 执行 -------------------- */
    int groupId = hthread_group_create(clusterId, nthreads);
    if (groupId == -1) { fprintf(stderr, "group create fail\n"); return 2; }

    uint64_t tDsp1=0, tDsp2=0, tDsp3=0, tCpu1=0, tCpu2=0, tCpu3=0;
    uint64_t st, ed;
    
    for (int t = 0; t < tmax; t++) {
        args1[2] = (uint64_t)t;
        args2[2] = (uint64_t)t;
        args3[2] = (uint64_t)t;
        
        st = getCurrentTimeMicros();
        hthread_group_exec(groupId, kernel1, 3, 6, args1);
        hthread_group_wait(groupId);
        ed = getCurrentTimeMicros();  tDsp1 += ed - st;

        st = getCurrentTimeMicros();
        hthread_group_exec(groupId, kernel2, 3, 5, args2);
        hthread_group_wait(groupId);
        ed = getCurrentTimeMicros();  tDsp2 += ed - st;

        st = getCurrentTimeMicros();
        hthread_group_exec(groupId, kernel3, 3, 5, args3);
        hthread_group_wait(groupId);
        ed = getCurrentTimeMicros();  tDsp3 += ed - st;
    }

    hthread_group_destroy(groupId);

    /* -------------------- CPU 执行 -------------------- */
    for (int t = 0; t < tmax; t++) {
        st = getCurrentTimeMicros();
        fdtd_cpu1(t, nx, ny, _fict_h, ex_h, ey_h, hz_h);
        ed = getCurrentTimeMicros();  tCpu1 += ed - st;

        st = getCurrentTimeMicros();
        fdtd_cpu2(t, nx, ny, _fict_h, ex_h, ey_h, hz_h);
        ed = getCurrentTimeMicros();  tCpu2 += ed - st;

        st = getCurrentTimeMicros();
        fdtd_cpu3(t, nx, ny, _fict_h, ex_h, ey_h, hz_h);
        ed = getCurrentTimeMicros();  tCpu3 += ed - st;
    }

    /* -------------------- 校验结果 ------------------- */
    int err = check_result(nx, ny, hz_h, hz_d);
    if (err != 0) {
        fprintf(stderr, "FDTD2D test FAILED!\n");
    } else {
        save_data("FDTD2D", tmax, nx, ny, before1, after1, tDsp1, tCpu1,
                  clusterId, devProgram, nthreads, kernel1);
        save_data("FDTD2D", tmax, nx, ny, before2, after2, tDsp2, tCpu2,
                  clusterId, devProgram, nthreads, kernel2);
        save_data("FDTD2D", tmax, nx, ny, before3, after3, tDsp3, tCpu3,
                  clusterId, devProgram, nthreads, kernel3);
        printf("WallTime FDTD2D_kernel1 (DSP/CPU): %fs / %fs\n",
                tDsp1/1e6, tCpu1/1e6);
        printf("WallTime FDTD2D_kernel2 (DSP/CPU): %fs / %fs\n",
                tDsp2/1e6, tCpu2/1e6);
        printf("WallTime FDTD2D_kernel3 (DSP/CPU): %fs / %fs\n",
                tDsp3/1e6, tCpu3/1e6);
    }

    /* -------------------- 资源释放 -------------------- */
    hthread_free(_fict_d); hthread_free(ex_d); hthread_free(ey_d); hthread_free(hz_d);
    hthread_free(before1); hthread_free(after1);
    hthread_free(before2); hthread_free(after2);
    hthread_free(before3); hthread_free(after3);

    free(_fict_h); free(ex_h); free(ey_h); free(hz_h);

    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return err ? 1 : 0;
}