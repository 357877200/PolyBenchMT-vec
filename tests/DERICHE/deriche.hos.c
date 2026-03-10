#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

#include "hthread_host.h"
#include "../common/tool.h"   // percentDiff(), getCurrentTimeMicros()

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/*************************************
 - CPU reference kernels (拆为6段)
************************************/

void deriche_cpu1(int w, int h, double a1, double a2, double b1, double b2,
                  double *imgIn, double *y1)
{
    for (int i = 0; i < w; i++) {
        double ym1 = 0.0;
        double ym2 = 0.0;
        double xm1 = 0.0;
        for (int j = 0; j < h; j++) {
            int idx = i*h + j;
            y1[idx] = a1 * imgIn[idx] + a2 * xm1 + b1 * ym1 + b2 * ym2;
            xm1 = imgIn[idx];
            ym2 = ym1;
            ym1 = y1[idx];
        }
    }
}

void deriche_cpu2(int w, int h, double a3, double a4, double b1, double b2,
                  double *imgIn, double *y2)
{
    for (int i = 0; i < w; i++) {
        double yp1 = 0.0, yp2 = 0.0;
        double xp1 = 0.0, xp2 = 0.0;
        for (int j = h-1; j >= 0; j--) {
            int idx = i*h + j;
            y2[idx] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
            xp2 = xp1;
            xp1 = imgIn[idx];
            yp2 = yp1;
            yp1 = y2[idx];
        }
    }
}

void deriche_cpu3(int w, int h, double c1,
                  double *y1, double *y2, double *imgOut)
{
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            int idx = i*h + j;
            imgOut[idx] = c1 * (y1[idx] + y2[idx]);
        }
    }
}

void deriche_cpu4(int w, int h, double a5, double a6, double b1, double b2,
                  double *imgOut, double *y1)
{
    for (int j = 0; j < h; j++) {
        double tm1 = 0.0;
        double ym1 = 0.0, ym2 = 0.0;
        for (int i = 0; i < w; i++) {
            int idx = i*h + j;
            y1[idx] = a5 * imgOut[idx] + a6 * tm1 + b1 * ym1 + b2 * ym2;
            tm1 = imgOut[idx];
            ym2 = ym1;
            ym1 = y1[idx];
        }
    }
}

// kernel5: 递推部分
void deriche_cpu5(int w, int h, double a7, double a8, double b1, double b2,
                  double *imgOut, double *y2)
{
    for (int j = 0; j < h; j++) {
        double tp1 = 0.0, tp2 = 0.0;
        double yp1 = 0.0, yp2 = 0.0;
        for (int i = w-1; i >= 0; i--) {
            int idx = i*h + j;
            y2[idx] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
            tp2 = tp1;
            tp1 = imgOut[idx];
            yp2 = yp1;
            yp1 = y2[idx];
        }
    }
}

// kernel6: 加法部分
void deriche_cpu6(int w, int h, double c2,
                  double *y1, double *y2, double *imgOut)
{
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            int idx = i*h + j;
            imgOut[idx] = c2 * (y1[idx] + y2[idx]);
        }
    }
}

/*************************************
 - data initialisation / check
************************************/
void init_array(int w, int h, double *alpha, double *imgIn, double *imgOut)
{
    *alpha = 0.25;
    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++) {
            imgIn[i*h + j] = ((313*i + 991*j) % 65536) / 65535.0;
            imgOut[i*h + j] = 0.0;
        }
}

int check_result(int n, double *ref, double *res)
{
    int errNum = 0;
    for (int i = 0; i < n; i++) {
        if (percentDiff(ref[i], res[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
            if (errNum < 10) {
                fprintf(stderr, "diff @[%d] : H=%.8f  D=%.8f\n", i, ref[i], res[i]);
            }
            errNum++;
        }
    }
    if (errNum) fprintf(stderr, "Total errors : %d\n", errNum);
    return errNum;
}

static void save_data(const char *bench,
                      int w, int h,
                      uint64_t *before, uint64_t *after,
                      uint64_t tDsp, uint64_t tCpu,
                      int clusterId, const char *program,
                      int nthreads, const char *kernel)
{
    FILE *fp = fopen("tests/DERICHE/deriche_events.txt", "a");
    if (!fp) { perror("fopen"); return; }
    fprintf(fp, "%s,%d,%d,%s,%s,%d,", bench, clusterId,
            w*h, program, kernel, nthreads);
    for (int i = 0; i < 26; i++) {
        fprintf(fp, "%lu", after[i] - before[i]);
        if (i != 25) fputc(',', fp);
    }
    fprintf(fp, ",%f,%f\n", tDsp/1e6, tCpu/1e6);
    fclose(fp);
}

/*************************************
 - main
************************************/
int main(int argc, char **argv)
{
    int clusterId = 1;
    int w = 64;
    int h = 64;
    int nthreads = 1;
    char *devProgram = "operators/DERICHE/deriche.dev.dat";
    char *kernel1 = "deriche_kernel1";
    char *kernel2 = "deriche_kernel2";
    char *kernel3 = "deriche_kernel3";
    char *kernel4 = "deriche_kernel4";
    char *kernel5 = "deriche_kernel5";
    char *kernel6 = "deriche_kernel6";

    for (int i = 1; i < argc; i++) {
        if (i+1 >= argc) break;
        if (!strcmp(argv[i], "--clusterId") || !strcmp(argv[i], "-c")) clusterId = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--w"))      w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--h"))      h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") || !strcmp(argv[i], "-t")) nthreads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--program") || !strcmp(argv[i], "-p")) devProgram = argv[++i];
        else if (!strcmp(argv[i], "--kernel1") || !strcmp(argv[i], "-k1")) kernel1 = argv[++i];
        else if (!strcmp(argv[i], "--kernel2") || !strcmp(argv[i], "-k2")) kernel2 = argv[++i];
        else if (!strcmp(argv[i], "--kernel3") || !strcmp(argv[i], "-k3")) kernel3 = argv[++i];
        else if (!strcmp(argv[i], "--kernel4") || !strcmp(argv[i], "-k4")) kernel4 = argv[++i];
        else if (!strcmp(argv[i], "--kernel5") || !strcmp(argv[i], "-k5")) kernel5 = argv[++i];
        else if (!strcmp(argv[i], "--kernel6") || !strcmp(argv[i], "-k6")) kernel6 = argv[++i];
    }

    if (clusterId < 0 || clusterId > 3) { fprintf(stderr, "invalid clusterId\n"); return 2; }
    if (nthreads <= 0) { fprintf(stderr, "invalid nthreads\n"); return 2; }
    if (access(devProgram, F_OK)) { fprintf(stderr, "%s not found\n", devProgram); return 2; }

    int retc;
    retc = hthread_dev_open(clusterId);  if (retc) { fprintf(stderr, "dev open fail\n"); return retc; }
    retc = hthread_dat_load(clusterId, devProgram);
    if (retc) { fprintf(stderr, "load dat fail\n"); return retc; }

    int avail = hthread_get_avail_threads(clusterId);
    if (nthreads > avail) {
        fprintf(stderr, "thread overflow: avail %d, ask %d\n", avail, nthreads);
        hthread_dat_unload(clusterId); hthread_dev_close(clusterId); return 2;
    }

    size_t sizeImg = (size_t)w * h * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    double *imgIn_d  = (double *)hthread_malloc(clusterId, sizeImg, HT_MEM_RO);
    double *imgOut_d = (double *)hthread_malloc(clusterId, sizeImg, HT_MEM_RW);
    double *y1_d     = (double *)hthread_malloc(clusterId, sizeImg, HT_MEM_RW);
    double *y2_d     = (double *)hthread_malloc(clusterId, sizeImg, HT_MEM_RW);

    uint64_t *before[6], *after[6];
    for (int k = 0; k < 6; k++) {
        before[k] = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
        after[k]  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
        if (!before[k] || !after[k]) { fprintf(stderr, "device malloc failed\n"); return 1; }
        memset(before[k], 0, sizeHot);
        memset(after[k], 0, sizeHot);
    }

    double *imgIn_h  = (double *)malloc(sizeImg);
    double *imgOut_h = (double *)malloc(sizeImg);
    double *y1_h     = (double *)malloc(sizeImg);
    double *y2_h     = (double *)malloc(sizeImg);
    double *imgOutRef_h = (double *)malloc(sizeImg);
    if (!imgIn_h || !imgOut_h || !y1_h || !y2_h || !imgOutRef_h) { fprintf(stderr, "malloc host fail\n"); return 1; }

    double alpha;
    init_array(w, h, &alpha, imgIn_h, imgOut_h);
    memcpy(imgIn_d, imgIn_h, sizeImg);
    memcpy(y1_d, y1_h, sizeImg);
    memcpy(y2_d, y2_h, sizeImg);
    memcpy(imgOut_d, imgOut_h, sizeImg);

    double k  = (1.0 - exp(-alpha)) * (1.0 - exp(-alpha)) /
                (1.0 + 2.0*exp(-alpha) - exp(-2.0*alpha));
    double a1 = k;
    double a2 = k * exp(-alpha) * (alpha - 1.0);
    double a3 = k * exp(-alpha) * (alpha + 1.0);
    double a4 = -k * exp(-2.0 * alpha);
    double b1 = pow(2.0, -alpha);
    double b2 = -exp(-2.0 * alpha);
    double c1 = 1.0;
    double a5 = a1;
    double a6 = a2;
    double a7 = a3;
    double a8 = a4;
    double c2 = 1.0;

    uint64_t args1[10], args2[10], args3[8], args4[10], args5[8], args6[6];
    args1[0]=w; args1[1]=h; args1[2]=doubleToRawBits(a1); args1[3]=doubleToRawBits(a2);
    args1[4]=doubleToRawBits(b1); args1[5]=doubleToRawBits(b2);
    args1[6]=(uint64_t)imgIn_d; args1[7]=(uint64_t)y1_d;
    args1[8]=(uint64_t)before[0]; args1[9]=(uint64_t)after[0];

    args2[0]=w; args2[1]=h; args2[2]=doubleToRawBits(a3); args2[3]=doubleToRawBits(a4);
    args2[4]=doubleToRawBits(b1); args2[5]=doubleToRawBits(b2);
    args2[6]=(uint64_t)imgIn_d; args2[7]=(uint64_t)y2_d;
    args2[8]=(uint64_t)before[1]; args2[9]=(uint64_t)after[1];

    args3[0]=w; args3[1]=h; args3[2]=doubleToRawBits(c1);
    args3[3]=(uint64_t)y1_d; args3[4]=(uint64_t)y2_d; args3[5]=(uint64_t)imgOut_d;
    args3[6]=(uint64_t)before[2]; args3[7]=(uint64_t)after[2];

    args4[0]=w; args4[1]=h; args4[2]=doubleToRawBits(a5); args4[3]=doubleToRawBits(a6);
    args4[4]=doubleToRawBits(b1); args4[5]=doubleToRawBits(b2);
    args4[6]=(uint64_t)imgOut_d; args4[7]=(uint64_t)y1_d;
    args4[8]=(uint64_t)before[3]; args4[9]=(uint64_t)after[3];

    args5[0]=w; args5[1]=h; args5[2]=doubleToRawBits(a7); args5[3]=doubleToRawBits(a8);
    args5[4]=doubleToRawBits(b1); args5[5]=doubleToRawBits(b2);
    args5[6]=(uint64_t)imgOut_d; args5[7]=(uint64_t)y2_d;

    args6[0]=w; args6[1]=h; args6[2]=doubleToRawBits(c2);
    args6[3]=(uint64_t)y1_d; args6[4]=(uint64_t)y2_d; args6[5]=(uint64_t)imgOut_d;

    int groupId = hthread_group_create(clusterId, nthreads);
    if (groupId == -1) { fprintf(stderr, "group create fail\n"); return 2; }

    uint64_t tDsp[6]={0}, tCpu[6]={0}, st, ed;

    // DSP 执行
    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel1, 6, 2, args1); hthread_group_wait(groupId);
    ed = getCurrentTimeMicros(); tDsp[0]+=ed-st;

    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel2, 6, 2, args2); hthread_group_wait(groupId);
    ed = getCurrentTimeMicros(); tDsp[1]+=ed-st;

    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel3, 3, 3, args3); hthread_group_wait(groupId);
    ed = getCurrentTimeMicros(); tDsp[2]+=ed-st;

    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel4, 6, 2, args4); hthread_group_wait(groupId);
    ed = getCurrentTimeMicros(); tDsp[3]+=ed-st;

    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel5, 6, 2, args5); hthread_group_wait(groupId);
    ed = getCurrentTimeMicros(); tDsp[4]+=ed-st;

    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel6, 3, 3, args6); hthread_group_wait(groupId);
    ed = getCurrentTimeMicros(); tDsp[5]+=ed-st;

    hthread_group_destroy(groupId);

    // CPU 执行
    st = getCurrentTimeMicros(); deriche_cpu1(w,h,a1,a2,b1,b2,imgIn_h,y1_h); ed = getCurrentTimeMicros(); tCpu[0]+=ed-st;
    st = getCurrentTimeMicros(); deriche_cpu2(w,h,a3,a4,b1,b2,imgIn_h,y2_h); ed = getCurrentTimeMicros(); tCpu[1]+=ed-st;
    st = getCurrentTimeMicros(); deriche_cpu3(w,h,c1,y1_h,y2_h,imgOutRef_h); ed = getCurrentTimeMicros(); tCpu[2]+=ed-st;
    st = getCurrentTimeMicros(); deriche_cpu4(w,h,a5,a6,b1,b2,imgOutRef_h,y1_h); ed = getCurrentTimeMicros(); tCpu[3]+=ed-st;
    st = getCurrentTimeMicros(); deriche_cpu5(w,h,a7,a8,b1,b2,imgOutRef_h,y2_h); ed = getCurrentTimeMicros(); tCpu[4]+=ed-st;
    st = getCurrentTimeMicros(); deriche_cpu6(w,h,c2,y1_h,y2_h,imgOutRef_h); ed = getCurrentTimeMicros(); tCpu[5]+=ed-st;

    // 校验结果
    memcpy(imgOut_h, imgOut_d, sizeImg);
    int err = check_result(w*h, imgOutRef_h, imgOut_h);
    for (int k = 0; k < 6; k++) {
        if (!err) {
            save_data("DERICHE", w,h, before[k], after[k], tDsp[k], tCpu[k],
                      clusterId, devProgram, nthreads,
                      (k==0)?kernel1:(k==1)?kernel2:(k==2)?kernel3:(k==3)?kernel4:(k==4)?kernel5:kernel6);
            printf("WallTime DERICHE_kernel%d (DSP/CPU): %fs / %fs\n", k+1, tDsp[k]/1e6, tCpu[k]/1e6);
        }
    }
    if (err) fprintf(stderr, "DERICHE test FAILED!\n");

    hthread_free(imgIn_d); hthread_free(imgOut_d); hthread_free(y1_d); hthread_free(y2_d);
    for (int k = 0; k < 6; k++) { hthread_free(before[k]); hthread_free(after[k]); }
    free(imgIn_h); free(imgOut_h); free(y1_h); free(y2_h); free(imgOutRef_h);

    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return err ? 1 : 0;
}