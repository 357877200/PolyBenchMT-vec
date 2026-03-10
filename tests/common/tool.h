#define SMALL_FLOAT_VAL 0.00000001f
float percentDiff(double val1, double val2)
{
    // 两个值都接近零时
    if ((fabs(val1) < 0.01) && (fabs(val2) < 0.01)) {
        return 0.0f;
    } else {
        // 更标准的百分比差异计算
        double max_val = (fabs(val1) > fabs(val2)) ? fabs(val1) : fabs(val2);
        if (max_val < SMALL_FLOAT_VAL)
            max_val = SMALL_FLOAT_VAL; // 防止除以零
        return 100.0f * (fabs(val1 - val2) / max_val);
    }
}
uint64_t doubleToRawBits(double d)
{
    union
    {
        uint64_t i;
        double f;
    } word;
    word.f = d;
    return word.i;
}


uint64_t getCurrentTimeMicros()
{
    struct timeval time;
    gettimeofday(&time, NULL);
    return (uint64_t)((time.tv_sec * INT64_C(1000000)) + time.tv_usec);
}

int fileIsExist(const char *filePath)
{
    return access(filePath, F_OK);
}

#define M_logError(_FMT, ...)                            \
    do                                                   \
    {                                                    \
        fprintf(stderr, "Error : " _FMT "in %d of %s\n", \
                __VA_ARGS__, __LINE__, __FILE__);        \
    } while (0);

#define M_checkRetC(_RETC, _MSG)                                    \
    do                                                              \
    {                                                               \
        if (_RETC != HT_SUCCESS)                                    \
        {                                                           \
            fprintf(stderr, "Failed to exec %s in %d of %s : %d\n", \
                    #_MSG, __LINE__, __FILE__, _RETC);              \
            return 2;                                               \
        }                                                           \
    } while (0);

#define M_checkMalloc(_PTR)                                      \
    do                                                           \
    {                                                            \
        if (_PTR == NULL)                                        \
        {                                                        \
            fprintf(stderr, "Failed to malloc %s in %d of %s\n", \
                    #_PTR, __LINE__, __FILE__);                  \
            return 2;                                            \
        }                                                        \
    } while (0);
