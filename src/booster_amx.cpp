#include "../include/config.h"

#if defined(__AMX_INT8__) && BOOST_AMX_SIEVE
#include "../include/pool_epi8.h"
#include "../include/bgj_amx.h"
#include "../include/bucket_amx.h"

#include <sys/time.h>
#include <sys/syscall.h>
#include <unistd.h>

#define TILE_DP160x2_TR(__ind)                              \
                               do {                         \
    _tile_zero(1);                                          \
    _tile_zero(2);                                          \
    _tile_loadd(4, _buf0_tr, 64);                           \
    _tile_loadd(5, _buf0 + ((__ind) + 16 * 0) * 64, 64);    \
    _tile_loadd(6, _buf0 + ((__ind) + 16 * 1) * 64, 64);    \
    _tile_dpbssd(1, 5, 4);                                  \
    _tile_loadd(7, _buf1_tr, 64);                           \
    _tile_loadd(3, _buf1 + ((__ind) + 16 * 0) * 64, 64);    \
    _tile_loadd(0, _buf1 + ((__ind) + 16 * 1) * 64, 64);    \
    _tile_dpbssd(2, 6, 4);                                  \
    _tile_loadd(5, _buf2_tr, 32);                           \
    _tile_dpbssd(1, 3, 7);                                  \
    _tile_loadd(6, _buf2 + ((__ind) + 16 * 0) * 64, 64);    \
    _tile_dpbssd(2, 0, 7);                                  \
    _tile_loadd(4, _buf2 + ((__ind) + 16 * 1) * 64, 64);    \
    _tile_dpbssd(1, 6, 5);                                  \
    _tile_stored(1, dst + 0 * 256, 64);                     \
    _tile_dpbssd(2, 4, 5);                                  \
    _tile_stored(2, dst + 1 * 256, 64);                     \
} while (0)

#define TILE_DP160x4_TR(__ind)                                  \
                                    do {                        \
    _tile_zero(0);                                              \
    _tile_zero(1);                                              \
    _tile_zero(2);                                              \
    _tile_zero(3);                                              \
    _tile_loadd(4, _buf0_tr, 64);                               \
    _tile_loadd(5, _buf0 + ((__ind) + 16 * 0) * 64, 64);        \
    _tile_loadd(6, _buf0 + ((__ind) + 16 * 1) * 64, 64);        \
    _tile_loadd(7, _buf0 + ((__ind) + 16 * 2) * 64, 64);        \
    _tile_dpbssd(0, 5, 4);                                      \
    _tile_loadd(5, _buf0 + ((__ind) + 16 * 3) * 64, 64);        \
    _tile_dpbssd(1, 6, 4);                                      \
    _tile_loadd(6, _buf1_tr, 64);                               \
    _tile_dpbssd(2, 7, 4);                                      \
    _tile_loadd(7, _buf1 + ((__ind) + 16 * 0) * 64, 64);        \
    _tile_dpbssd(3, 5, 4);                                      \
    _tile_loadd(4, _buf1 + ((__ind) + 16 * 1) * 64, 64);        \
    _tile_loadd(5, _buf1 + ((__ind) + 16 * 2) * 64, 64);        \
    _tile_dpbssd(0, 7, 6);                                      \
    _tile_loadd(7, _buf1 + ((__ind) + 16 * 3) * 64, 64);        \
    _tile_dpbssd(1, 4, 6);                                      \
    _tile_loadd(4, _buf2_tr, 32);                               \
    _tile_dpbssd(2, 5, 6);                                      \
    _tile_loadd(5, _buf2 + ((__ind) + 16 * 0) * 64, 64);        \
    _tile_dpbssd(3, 7, 6);                                      \
    _tile_loadd(6, _buf2 + ((__ind) + 16 * 1) * 64, 64);        \
    _tile_loadd(7, _buf2 + ((__ind) + 16 * 2) * 64, 64);        \
    _tile_dpbssd(0, 5, 4);                                      \
    _tile_loadd(5, _buf2 + ((__ind) + 16 * 3) * 64, 64);        \
    _tile_stored(0, dst + 0 * 256, 64);                         \
    _tile_dpbssd(1, 6, 4);                                      \
    _tile_stored(1, dst + 1 * 256, 64);                         \
    _tile_dpbssd(2, 7, 4);                                      \
    _tile_stored(2, dst + 2 * 256, 64);                         \
    _tile_dpbssd(3, 5, 4);                                      \
    _tile_stored(3, dst + 3 * 256, 64);                         \
} while (0)


#define COMPUTE_VEC_AND_NORM_B8(__dst, __norm_dst, __ind_src, __p) \
                                                            do {    \
    __m512i all0x00 = _mm512_setzero_si512();   \
    __m512i a0, a1, a2, a3, a4, a5, a6, a7; \
    __m512i acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7; \
    __m512i sacc0, sacc1, sacc2, sacc3, sacc4, sacc5, sacc6, sacc7; \
    __m256i b0, b1, b2, b3, b4, b5, b6, b7; \
    __m256i sacc0_256, sacc1_256, sacc2_256, sacc3_256, sacc4_256, sacc5_256, sacc6_256, sacc7_256;\
    if (__p) {  \
        a0 = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[0])),   \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[1])));  \
        a1 = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[2])),  \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[3])));  \
        a2 = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[4])),  \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[5])));  \
        a3 = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[6])),  \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[7])));  \
        a4 = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[8])),  \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[9])));  \
        a5 = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[10])), \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[11]))); \
        a6 = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[12])), \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[13]))); \
        a7 = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[14])), \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[15]))); \
    } else {    \
        a0 = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[0])),  \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[1])));  \
        a1 = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[2])),  \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[3])));  \
        a2 = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[4])),  \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[5])));  \
        a3 = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[6])),  \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[7])));  \
        a4 = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[8])),  \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[9])));  \
        a5 = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[10])), \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[11]))); \
        a6 = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[12])), \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[13]))); \
        a7 = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[14])), \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[15]))); \
    }       \
    _mm512_storeu_si512((__m512 *)(__dst), a0); \
    _mm512_storeu_si512((__m512 *)((__dst) + 160), a1); \
    _mm512_storeu_si512((__m512 *)((__dst) + 320), a2); \
    _mm512_storeu_si512((__m512 *)((__dst) + 480), a3); \
    _mm512_storeu_si512((__m512 *)((__dst) + 640), a4); \
    _mm512_storeu_si512((__m512 *)((__dst) + 800), a5); \
    _mm512_storeu_si512((__m512 *)((__dst) + 960), a6); \
    _mm512_storeu_si512((__m512 *)((__dst) + 1120), a7);\
    acc0 = _mm512_dpbusd_epi32(all0x00, _mm512_xor_si512(a0, all0x80), a0); \
    acc1 = _mm512_dpbusd_epi32(all0x00, _mm512_xor_si512(a1, all0x80), a1); \
    acc2 = _mm512_dpbusd_epi32(all0x00, _mm512_xor_si512(a2, all0x80), a2); \
    acc3 = _mm512_dpbusd_epi32(all0x00, _mm512_xor_si512(a3, all0x80), a3); \
    acc4 = _mm512_dpbusd_epi32(all0x00, _mm512_xor_si512(a4, all0x80), a4); \
    acc5 = _mm512_dpbusd_epi32(all0x00, _mm512_xor_si512(a5, all0x80), a5); \
    acc6 = _mm512_dpbusd_epi32(all0x00, _mm512_xor_si512(a6, all0x80), a6); \
    acc7 = _mm512_dpbusd_epi32(all0x00, _mm512_xor_si512(a7, all0x80), a7); \
    sacc0 = _mm512_dpbusd_epi32(all0x00, all0x80, a0);  \
    sacc1 = _mm512_dpbusd_epi32(all0x00, all0x80, a1);  \
    sacc2 = _mm512_dpbusd_epi32(all0x00, all0x80, a2);  \
    sacc3 = _mm512_dpbusd_epi32(all0x00, all0x80, a3);  \
    sacc4 = _mm512_dpbusd_epi32(all0x00, all0x80, a4);  \
    sacc5 = _mm512_dpbusd_epi32(all0x00, all0x80, a5);  \
    sacc6 = _mm512_dpbusd_epi32(all0x00, all0x80, a6);  \
    sacc7 = _mm512_dpbusd_epi32(all0x00, all0x80, a7);  \
    if (__p) {  \
        a0 = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[0])),   \
                            _mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[1])));     \
        a1 = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[2])), \
                            _mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[3])));\
        a2 = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[4])), \
                            _mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[5]))); \
        a3 = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[6])), \
                            _mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[7]))); \
        a4 = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[8])), \
                            _mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[9]))); \
        a5 = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[10])),    \
                            _mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[11])));    \
        a6 = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[12])),    \
                            _mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[13])));    \
        a7 = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[14])),    \
                            _mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[15])));    \
    } else {    \
        a0 = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[0])),\
                            _mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[1]))); \
        a1 = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[2])), \
                            _mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[3]))); \
        a2 = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[4])), \
                            _mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[5]))); \
        a3 = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[6])), \
                            _mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[7]))); \
        a4 = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[8])), \
                            _mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[9]))); \
        a5 = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[10])),    \
                            _mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[11])));    \
        a6 = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[12])),    \
                            _mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[13])));    \
        a7 = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[14])),    \
                            _mm512_loadu_si512(p->vec + 64 + 160ULL * (long) ((__ind_src)[15])));    \
    }   \
    _mm512_storeu_si512((__m512 *)(__dst) + 1, a0); \
    _mm512_storeu_si512((__m512 *)((__dst) + 160) + 1, a1); \
    _mm512_storeu_si512((__m512 *)((__dst) + 320) + 1, a2); \
    _mm512_storeu_si512((__m512 *)((__dst) + 480) + 1, a3); \
    _mm512_storeu_si512((__m512 *)((__dst) + 640) + 1, a4); \
    _mm512_storeu_si512((__m512 *)((__dst) + 800) + 1, a5); \
    _mm512_storeu_si512((__m512 *)((__dst) + 960) + 1, a6); \
    _mm512_storeu_si512((__m512 *)((__dst) + 1120) + 1, a7);    \
    acc0 = _mm512_sub_epi32(_mm512_dpbusd_epi32(acc0, _mm512_xor_si512(a0, all0x80), a0), _mm512_dpbusd_epi32(sacc0, all0x80, a0));\
    acc1 = _mm512_sub_epi32(_mm512_dpbusd_epi32(acc1, _mm512_xor_si512(a1, all0x80), a1), _mm512_dpbusd_epi32(sacc1, all0x80, a1)); \
    acc2 = _mm512_sub_epi32(_mm512_dpbusd_epi32(acc2, _mm512_xor_si512(a2, all0x80), a2), _mm512_dpbusd_epi32(sacc2, all0x80, a2)); \
    acc3 = _mm512_sub_epi32(_mm512_dpbusd_epi32(acc3, _mm512_xor_si512(a3, all0x80), a3), _mm512_dpbusd_epi32(sacc3, all0x80, a3)); \
    acc4 = _mm512_sub_epi32(_mm512_dpbusd_epi32(acc4, _mm512_xor_si512(a4, all0x80), a4), _mm512_dpbusd_epi32(sacc4, all0x80, a4)); \
    acc5 = _mm512_sub_epi32(_mm512_dpbusd_epi32(acc5, _mm512_xor_si512(a5, all0x80), a5), _mm512_dpbusd_epi32(sacc5, all0x80, a5)); \
    acc6 = _mm512_sub_epi32(_mm512_dpbusd_epi32(acc6, _mm512_xor_si512(a6, all0x80), a6), _mm512_dpbusd_epi32(sacc6, all0x80, a6)); \
    acc7 = _mm512_sub_epi32(_mm512_dpbusd_epi32(acc7, _mm512_xor_si512(a7, all0x80), a7), _mm512_dpbusd_epi32(sacc7, all0x80, a7)); \
    \
    __m256i acc0_256 = _mm256_add_epi32(_mm512_castsi512_si256(acc0), _mm512_extracti32x8_epi32(acc0, 1));  \
    __m256i acc1_256 = _mm256_add_epi32(_mm512_castsi512_si256(acc1), _mm512_extracti32x8_epi32(acc1, 1));  \
    __m256i acc2_256 = _mm256_add_epi32(_mm512_castsi512_si256(acc2), _mm512_extracti32x8_epi32(acc2, 1));  \
    __m256i acc3_256 = _mm256_add_epi32(_mm512_castsi512_si256(acc3), _mm512_extracti32x8_epi32(acc3, 1));  \
    __m256i acc4_256 = _mm256_add_epi32(_mm512_castsi512_si256(acc4), _mm512_extracti32x8_epi32(acc4, 1));  \
    __m256i acc5_256 = _mm256_add_epi32(_mm512_castsi512_si256(acc5), _mm512_extracti32x8_epi32(acc5, 1));  \
    __m256i acc6_256 = _mm256_add_epi32(_mm512_castsi512_si256(acc6), _mm512_extracti32x8_epi32(acc6, 1));  \
    __m256i acc7_256 = _mm256_add_epi32(_mm512_castsi512_si256(acc7), _mm512_extracti32x8_epi32(acc7, 1));  \
    \
    if (__p) {  \
        b0 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[0]))),   \
                            _mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[1]))));    \
        b1 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[2]))),    \
                            _mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[3]))));    \
        b2 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[4]))),    \
                            _mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[5]))));    \
        b3 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[6]))),    \
                            _mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[7]))));    \
        b4 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[8]))),    \
                            _mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[9]))));    \
        b5 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[10]))),   \
                            _mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[11]))));   \
        b6 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[12]))),   \
                            _mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[13]))));   \
        b7 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[14]))),   \
                            _mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[15]))));   \
    } else {    \
        b0 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[0]))),    \
                            _mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[1]))));    \
        b1 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[2]))),    \
                            _mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[3]))));    \
        b2 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[4]))),    \
                            _mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[5]))));    \
        b3 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[6]))),    \
                            _mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[7]))));    \
        b4 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[8]))),    \
                            _mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[9]))));    \
        b5 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[10]))),   \
                            _mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[11]))));   \
        b6 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[12]))),   \
                            _mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[13]))));   \
        b7 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[14]))),   \
                            _mm256_load_si256((__m256i *)(p->vec + 128 + 160ULL * (long) ((__ind_src)[15]))));   \
    }   \
    \
    _mm256_store_si256((__m256i *)(__dst) + 4, b0); \
    _mm256_store_si256((__m256i *)((__dst) + 160) + 4, b1);\
    _mm256_store_si256((__m256i *)((__dst) + 320) + 4, b2); \
    _mm256_store_si256((__m256i *)((__dst) + 480) + 4, b3); \
    _mm256_store_si256((__m256i *)((__dst) + 640) + 4, b4); \
    _mm256_store_si256((__m256i *)((__dst) + 800) + 4, b5); \
    _mm256_store_si256((__m256i *)((__dst) + 960) + 4, b6); \
    _mm256_store_si256((__m256i *)((__dst) + 1120) + 4, b7);    \
    \
    sacc0_256 = _mm256_dpbusd_epi32(_mm512_castsi512_si256(all0x00), _mm512_castsi512_si256(all0x80), b0);\
    sacc1_256 = _mm256_dpbusd_epi32(_mm512_castsi512_si256(all0x00), _mm512_castsi512_si256(all0x80), b1);  \
    sacc2_256 = _mm256_dpbusd_epi32(_mm512_castsi512_si256(all0x00), _mm512_castsi512_si256(all0x80), b2);  \
    sacc3_256 = _mm256_dpbusd_epi32(_mm512_castsi512_si256(all0x00), _mm512_castsi512_si256(all0x80), b3);  \
    sacc4_256 = _mm256_dpbusd_epi32(_mm512_castsi512_si256(all0x00), _mm512_castsi512_si256(all0x80), b4);  \
    sacc5_256 = _mm256_dpbusd_epi32(_mm512_castsi512_si256(all0x00), _mm512_castsi512_si256(all0x80), b5);  \
    sacc6_256 = _mm256_dpbusd_epi32(_mm512_castsi512_si256(all0x00), _mm512_castsi512_si256(all0x80), b6);  \
    sacc7_256 = _mm256_dpbusd_epi32(_mm512_castsi512_si256(all0x00), _mm512_castsi512_si256(all0x80), b7);  \
    acc0_256 = _mm256_sub_epi32(_mm256_dpbusd_epi32(acc0_256, _mm256_xor_si256(b0, _mm512_castsi512_si256(all0x80)), b0), sacc0_256);\
    acc1_256 = _mm256_sub_epi32(_mm256_dpbusd_epi32(acc1_256, _mm256_xor_si256(b1, _mm512_castsi512_si256(all0x80)), b1), sacc1_256);\
    acc2_256 = _mm256_sub_epi32(_mm256_dpbusd_epi32(acc2_256, _mm256_xor_si256(b2, _mm512_castsi512_si256(all0x80)), b2), sacc2_256);\
    acc3_256 = _mm256_sub_epi32(_mm256_dpbusd_epi32(acc3_256, _mm256_xor_si256(b3, _mm512_castsi512_si256(all0x80)), b3), sacc3_256);\
    acc4_256 = _mm256_sub_epi32(_mm256_dpbusd_epi32(acc4_256, _mm256_xor_si256(b4, _mm512_castsi512_si256(all0x80)), b4), sacc4_256);\
    acc5_256 = _mm256_sub_epi32(_mm256_dpbusd_epi32(acc5_256, _mm256_xor_si256(b5, _mm512_castsi512_si256(all0x80)), b5), sacc5_256);\
    acc6_256 = _mm256_sub_epi32(_mm256_dpbusd_epi32(acc6_256, _mm256_xor_si256(b6, _mm512_castsi512_si256(all0x80)), b6), sacc6_256);\
    acc7_256 = _mm256_sub_epi32(_mm256_dpbusd_epi32(acc7_256, _mm256_xor_si256(b7, _mm512_castsi512_si256(all0x80)), b7), sacc7_256);\
    acc0_256 = _mm256_hadd_epi32(acc0_256, acc1_256);\
    acc2_256 = _mm256_hadd_epi32(acc2_256, acc3_256);\
    acc4_256 = _mm256_hadd_epi32(acc4_256, acc5_256);\
    acc6_256 = _mm256_hadd_epi32(acc6_256, acc7_256);\
    acc0_256 = _mm256_hadd_epi32(acc0_256, acc2_256);\
    acc4_256 = _mm256_hadd_epi32(acc4_256, acc6_256);\
    __m256i acclo = _mm256_permute2f128_si256(acc0_256, acc4_256, 48);\
    __m256i acchi = _mm256_permute2f128_si256(acc0_256, acc4_256, 33);\
    _mm256_store_si256((__m256i *)(__norm_dst), _mm256_srai_epi32(_mm256_add_epi32(acclo, acchi), 1));\
} while (0)

#define COMPUTE_VEC_AND_NORM(__dst, __norm_dst, __ind_src, __p)     \
                                                            do {    \
    __m512i a, b;   \
    __m256i c;  \
    __m512i acc, sacc;  \
    if (__p) {  \
        a = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[0])),   \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[1])));  \
        b = _mm512_add_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[0]) + 64),  \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[1]) + 64)); \
        c = _mm256_add_epi8(_mm256_load_si256((__m256i *)(p->vec + 160ULL * (long) ((__ind_src)[0]) + 128)),    \
                            _mm256_load_si256((__m256i *)(p->vec + 160ULL * (long) ((__ind_src)[1]) + 128)));   \
    } else {    \
        a = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[0])),   \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[1])));  \
        b = _mm512_sub_epi8(_mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[0]) + 64),  \
                            _mm512_loadu_si512(p->vec + 160ULL * (long) ((__ind_src)[1]) + 64)); \
        c = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(p->vec + 160ULL * (long) ((__ind_src)[0]) + 128)),    \
                            _mm256_load_si256((__m256i *)(p->vec + 160ULL * (long) ((__ind_src)[1]) + 128)));   \
    }   \
    _mm512_storeu_si512((__m512i *)(__dst), a);  \
    _mm512_storeu_si512((__m512i *)(__dst) + 1, b);  \
    _mm256_store_si256((__m256i *)(__dst) + 4, c);  \
    acc = _mm512_setzero_si512();   \
    sacc = _mm512_setzero_si512();  \
    acc = _mm512_dpbusd_epi32(acc, _mm512_xor_si512(a, all0x80), a);  \
    acc = _mm512_dpbusd_epi32(acc, _mm512_xor_si512(b, all0x80), b);  \
    sacc = _mm512_dpbusd_epi32(sacc, all0x80, a);    \
    sacc = _mm512_dpbusd_epi32(sacc, all0x80, b);    \
    __m256i acc256 = _mm256_add_epi32(_mm512_castsi512_si256(acc), _mm512_extracti32x8_epi32(acc, 1));  \
    __m256i sacc256 = _mm256_add_epi32(_mm512_castsi512_si256(sacc), _mm512_extracti32x8_epi32(sacc, 1));   \
    acc256 = _mm256_dpbusd_epi32(acc256, _mm256_xor_si256(c, _mm512_castsi512_si256(all0x80)), c);    \
    sacc256 = _mm256_dpbusd_epi32(sacc256, _mm512_castsi512_si256(all0x80), c); \
    acc256 = _mm256_sub_epi32(acc256, sacc256); \
    __m128i acc128 = _mm_add_epi32(_mm256_castsi256_si128(acc256), _mm256_extracti128_si256(acc256, 1));    \
    acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, 78));  \
    acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, 177)); \
    *(__norm_dst) = _mm_cvtsi128_si32(acc128) >> 1;  \
} while (0)


booster_amx160_t::~booster_amx160_t() {
    if (lfp_ps16) FREE_MAT((void **)lfp_ps16);
    if (_coeff) FREE_VEC((void *)_coeff);
    if (_score_ps16) FREE_VEC((void *)_score_ps16);
    if (_norm_ps16) FREE_VEC((void *)_norm_ps16);
    if (_fvec_ps16) FREE_MAT((void **)_fvec_ps16);
}

int booster_amx160_t::init(Pool_epi8_t<5> *_p, long _esd, double _prefer_deep) {
    if (syscall(SYS_arch_prctl, 0x1023, 18)) {
        printf("[Error] booster_amx160_t::init: Fail to do XFEATURE_XTILEDATA, nothing done.\n");
        return -1;
    }

    if (_esd == -1) _esd = _p->index_l > AMX_BOOST_MAX_DIM ? AMX_BOOST_MAX_DIM : _p->index_l;
    double _mask_ratio = 1.0;
    long num_msk = 0;
    if (_prefer_deep <= -1.0) {
        _mask_ratio = - _prefer_deep;
        num_msk = round((_mask_ratio - 1.0) / AMX_BOOST_DOWNSIEVE_MASK_RATIO); 
        _prefer_deep = AMX_BOOST_PREFER_DEEP;
    }

    p = _p;
    ESD = _esd;
    CSD = p->CSD;

    if (lfp_ps16) FREE_MAT((void **)lfp_ps16);
    lfp_ps16 = (__m512 **) NEW_MAT(CSD + ESD, ESD, 512);

    if (!_coeff) _coeff = (float *) NEW_VEC(AMX_MAX_NTHREADS * 10240, sizeof(float));
    if (!_score_ps16) _score_ps16 = (__m512 *) NEW_VEC(AMX_MAX_NTHREADS * 4, sizeof(__m512));
    if (!_norm_ps16) _norm_ps16 = (__m512 *) NEW_VEC(AMX_MAX_NTHREADS * 4, sizeof(__m512));
    if (!_fvec_ps16) _fvec_ps16 = (__m512 **) NEW_MAT(AMX_MAX_NTHREADS * AMX_BOOST_MAX_DIM, 4, 512);
    
    iratio_ps16 = _mm512_set1_ps(2.0 / p->_ratio / p->_ratio);
    dhalf_si512 = _mm512_set1_epi32(p->_dhalf);
    dshift = p->_dshift;
    Lattice_QP *L = p->basis->b_loc_QP(p->index_l - ESD, p->index_r);
    Lattice_QP *L_loc_target = new Lattice_QP(CSD, CSD);
    for (long i = 0; i < CSD; i++) {
        for (long j = 0; j < CSD; j++) L_loc_target->get_b().hi[i][j] = p->_b_local[i][j] / p->_ratio;
    }
    L->trans_to(ESD, ESD + CSD, L_loc_target);
    for (long i = ESD; i < ESD + CSD; i++) {
        for (long j = ESD - 1; j >= 0; j--) {
            long q = round(L->get_b().hi[i][j] / L->get_b().hi[j][j]);
            red(L->get_b().hi[i], L->get_b().lo[i], L->get_b().hi[j], L->get_b().lo[j], NTL::quad_float(q), ESD);
        }
    }
    delete L_loc_target;
    for (long i = 0; i < ESD; i++) {
        for (long j = 0; j <= i; j++) {
            lfp_ps16[i][j] = _mm512_set1_ps(L->get_b().hi[i][j]);
        }
    }
    for (long i = ESD; i < ESD + CSD; i++) {
        for (long j = 0; j < ESD; j++) {
            lfp_ps16[i][j] = _mm512_set1_ps(L->get_b().hi[i][j]);
        }
    }
    for (long i = 0; i < ESD; i++) {
        double x = p->_ratio / L->gh(i, L->NumRows());
        x = pow(AMX_BOOST_PREFER_DEEP, i - ESD) * 0.5 * x * x * p->gh2;
        if (AMX_BOOST_DOWNSIEVE_MASK_DIM + i >= ESD) {
            if (AMX_BOOST_DOWNSIEVE_MASK_DIM + i < ESD + num_msk) x *= _mask_ratio;
        }            
        ighc_ps16[i] = _mm512_set1_ps(x);
        inorm_ps16[i] = _mm512_set1_ps(1.0 / L->get_b().hi[i][i]);
    }
    delete L;

    const __m512i all0x80 = _mm512_set1_epi8(0x80);
    const __m512i all0xff = _mm512_set1_epi8(0xff);
    const __m512i all0x00 = _mm512_set1_epi8(0x00);
    uint64_t m0, m1;
    uint32_t m2;
    if (CSD >= 128) {
        m0 = (-1ULL);
        m1 = (-1ULL);
        m2 = (1ULL << (CSD - 128)) - 1;
    } else if (CSD >= 64) {
        m0 = (-1ULL);
        m1 = (1ULL << (CSD - 64)) - 1;
        m2 = 0;
    } else {
        m0 = (1ULL << CSD) - 1;
        m1 = 0;
        m2 = 0;
    }
    const __m512i dmsk0 = _mm512_mask_mov_epi8(all0x00, m0, all0xff);
    const __m512i dmsk1 = _mm512_mask_mov_epi8(all0x00, m1, all0xff);
    const __m256i dmsk2 = _mm256_mask_mov_epi8(_mm512_castsi512_si256(all0x00), m2, _mm512_castsi512_si256(all0xff));
    for (long i = 0; i < CSD; i++) {
        _mm512_store_si512(&b_dual_0[i*64], _mm512_and_si512(dmsk0, _mm512_xor_si512(_mm512_loadu_si512(&(p->_b_dual[i * p->vec_length])), all0x80)));
        _mm512_store_si512(&b_dual_1[i*64], _mm512_and_si512(dmsk1, _mm512_xor_si512(_mm512_loadu_si512(&(p->_b_dual[i * p->vec_length + 64])), all0x80)));
        _mm256_store_si256((__m256i *)(&b_dual_2[i*64]), _mm256_and_si256(dmsk2, _mm256_xor_si256(
            _mm256_loadu_si256((__m256i *)(&(p->_b_dual[i * p->vec_length + 128]))), _mm512_castsi512_si256(all0x80))));
    }

    return 0;
}

void booster_amx160_t::_process_block64(int8_t *src, int32_t *src_norm, long thread) {
    #if 0
    // for debug
    __attribute__ ((aligned (64))) int32_t dbg_coeff[160];
    __attribute__ ((aligned (64))) float dbg_fvec[AMX_BOOST_MAX_DIM];
    float **dbg_lfp = (float **) NEW_MAT(CSD + ESD, ESD, sizeof(float));
    do {
        Lattice_QP *L = p->basis->b_loc_QP(p->index_l - ESD, p->index_r);
        Lattice_QP *L_loc_target = new Lattice_QP(CSD, CSD);
        for (long i = 0; i < CSD; i++) {
            for (long j = 0; j < CSD; j++) L_loc_target->get_b().hi[i][j] = p->_b_local[i][j] / p->_ratio;
        }
        L->trans_to(ESD, ESD + CSD, L_loc_target);
        delete L_loc_target;
        for (long i = 0; i < ESD; i++) {
            for (long j = 0; j <= i; j++) {
                dbg_lfp[i][j] = L->get_b().hi[i][j];
            }
        }
        for (long i = ESD; i < ESD + CSD; i++) {
            for (long j = 0; j < ESD; j++) {
                dbg_lfp[i][j] = L->get_b().hi[i][j];
            }
        }
        delete L;
    } while (0);
    for (long i = 0; i < 64; i++) {
        p->_compute_coeff(dbg_coeff, src + i * 160, p->_compute_sum(src + i * 160));
        set_zero_avx2(dbg_fvec, AMX_BOOST_MAX_DIM);
        for (long j = 0; j < CSD; j++) {
            red_avx2(dbg_fvec, dbg_lfp[ESD+j], -dbg_coeff[j], AMX_BOOST_MAX_DIM);
        }
        // PRINT_VEC(dbg_fvec, AMX_BOOST_MAX_DIM);
        float nn = src_norm[i] * (*(float*)(&iratio_ps16));
        float sc = src_norm[i];
        // printf("%d %d\t", (int)sqrt(nn), (int)(sc));
        for (long j = ESD - 1; j >= 0; j--) {
            float q = round(dbg_fvec[j] * (*(float *)(&inorm_ps16[j])));
            red_avx2(dbg_fvec, dbg_lfp[j], q, AMX_BOOST_MAX_DIM);
            nn += dbg_fvec[j] * dbg_fvec[j];
            sc = sc < nn * (*(float*)(&ighc_ps16[j])) ? sc : nn * (*(float*)(&ighc_ps16[j]));
            // printf("%d %d\t", (int)sqrt(nn), (int)(sc));
        }
        printf("%d, %d, ", (int)sqrt(nn), (int) sc);
        // PRINT_VEC(dbg_fvec, AMX_BOOST_MAX_DIM);
        // printf("\n\n");
        ((float *)(&score_ps16[0]))[i] = sc;
    }
    FREE_MAT(dbg_lfp);
    #else
    float *coeff = _coeff + thread * 10240;
    __m512 *score_ps16 = _score_ps16 + thread * 4;
    __m512 *norm_ps16 = _norm_ps16 + thread * 4;
    __m512 **fvec_ps16 = _fvec_ps16 + thread * AMX_BOOST_MAX_DIM;

    /////// STEP: compute coeff from src ///////
    int32_t *coeff_epi32 = (int32_t *)coeff;
    __attribute__ ((aligned (64))) int8_t _buf0_tr[1024];
    __attribute__ ((aligned (64))) int8_t _buf1_tr[1024];
    __attribute__ ((aligned (64))) int8_t _buf2_tr[512+32] = {};
    for (long ind = 0; ind < 64; ind += 16) {
        __m512i z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, za, zb, zc, zd, ze, zf;
        z0 = _mm512_loadu_si512(src + (ind + 0) * 160);
        z1 = _mm512_loadu_si512(src + (ind + 1) * 160);
        z2 = _mm512_loadu_si512(src + (ind + 2) * 160);
        z3 = _mm512_loadu_si512(src + (ind + 3) * 160);
        z4 = _mm512_loadu_si512(src + (ind + 4) * 160);
        z5 = _mm512_loadu_si512(src + (ind + 5) * 160);
        z6 = _mm512_loadu_si512(src + (ind + 6) * 160);
        z7 = _mm512_loadu_si512(src + (ind + 7) * 160);
        z8 = _mm512_loadu_si512(src + (ind + 8) * 160);
        z9 = _mm512_loadu_si512(src + (ind + 9) * 160);
        za = _mm512_loadu_si512(src + (ind + 10) * 160);
        zb = _mm512_loadu_si512(src + (ind + 11) * 160);
        zc = _mm512_loadu_si512(src + (ind + 12) * 160);
        zd = _mm512_loadu_si512(src + (ind + 13) * 160);
        ze = _mm512_loadu_si512(src + (ind + 14) * 160);
        zf = _mm512_loadu_si512(src + (ind + 15) * 160);
        AVX512_MATTR_16x16(NULL, _buf0_tr, z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, za, zb, zc, zd, ze, zf);
        
        z0 = _mm512_loadu_si512(src + (ind + 0) * 160 + 64);
        z1 = _mm512_loadu_si512(src + (ind + 1) * 160 + 64);
        z2 = _mm512_loadu_si512(src + (ind + 2) * 160 + 64);
        z3 = _mm512_loadu_si512(src + (ind + 3) * 160 + 64);
        z4 = _mm512_loadu_si512(src + (ind + 4) * 160 + 64);
        z5 = _mm512_loadu_si512(src + (ind + 5) * 160 + 64);
        z6 = _mm512_loadu_si512(src + (ind + 6) * 160 + 64);
        z7 = _mm512_loadu_si512(src + (ind + 7) * 160 + 64);
        z8 = _mm512_loadu_si512(src + (ind + 8) * 160 + 64);
        z9 = _mm512_loadu_si512(src + (ind + 9) * 160 + 64);
        za = _mm512_loadu_si512(src + (ind + 10) * 160 + 64);
        zb = _mm512_loadu_si512(src + (ind + 11) * 160 + 64);
        zc = _mm512_loadu_si512(src + (ind + 12) * 160 + 64);
        zd = _mm512_loadu_si512(src + (ind + 13) * 160 + 64);
        ze = _mm512_loadu_si512(src + (ind + 14) * 160 + 64);
        zf = _mm512_loadu_si512(src + (ind + 15) * 160 + 64);
        AVX512_MATTR_16x16(NULL, _buf1_tr, z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, za, zb, zc, zd, ze, zf);

        __m256i y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, ya, yb, yc, yd, ye, yf;
        y0 = _mm256_load_si256((__m256i *)(src + (ind + 0) * 160 + 128));
        y1 = _mm256_load_si256((__m256i *)(src + (ind + 1) * 160 + 128));
        y2 = _mm256_load_si256((__m256i *)(src + (ind + 2) * 160 + 128));
        y3 = _mm256_load_si256((__m256i *)(src + (ind + 3) * 160 + 128));
        y4 = _mm256_load_si256((__m256i *)(src + (ind + 4) * 160 + 128));
        y5 = _mm256_load_si256((__m256i *)(src + (ind + 5) * 160 + 128));
        y6 = _mm256_load_si256((__m256i *)(src + (ind + 6) * 160 + 128));
        y7 = _mm256_load_si256((__m256i *)(src + (ind + 7) * 160 + 128));
        y8 = _mm256_load_si256((__m256i *)(src + (ind + 8) * 160 + 128));
        y9 = _mm256_load_si256((__m256i *)(src + (ind + 9) * 160 + 128));
        ya = _mm256_load_si256((__m256i *)(src + (ind + 10) * 160 + 128));
        yb = _mm256_load_si256((__m256i *)(src + (ind + 11) * 160 + 128));
        yc = _mm256_load_si256((__m256i *)(src + (ind + 12) * 160 + 128));
        yd = _mm256_load_si256((__m256i *)(src + (ind + 13) * 160 + 128));
        ye = _mm256_load_si256((__m256i *)(src + (ind + 14) * 160 + 128));
        yf = _mm256_load_si256((__m256i *)(src + (ind + 15) * 160 + 128));
        AVX512_MATTR_16x8(NULL, _buf2_tr, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, ya, yb, yc, yd, ye, yf);

        int8_t *_buf0 = b_dual_0;
        int8_t *_buf1 = b_dual_1;
        int8_t *_buf2 = b_dual_2;
        int32_t *dst = coeff_epi32 + ind * 160;
        TILE_DP160x4_TR(0);
        dst += 1024;
        TILE_DP160x4_TR(64);
        dst += 1024;
        TILE_DP160x2_TR(128);
        dst += 512;
    }    
    for (long ind = 0; ind < 64; ind += 16) {
        for (long i = 0; i < CSD; i+=8) {
            float *ptr = coeff + ind * 160 + i * 16;
            _mm512_store_ps(ptr, _mm512_cvtepi32_ps(_mm512_srai_epi32(_mm512_add_epi32(dhalf_si512, _mm512_load_si512(ptr)), dshift)));
            _mm512_store_ps(ptr + 16, _mm512_cvtepi32_ps(_mm512_srai_epi32(_mm512_add_epi32(dhalf_si512, _mm512_load_si512(ptr + 16)), dshift)));
            _mm512_store_ps(ptr + 32, _mm512_cvtepi32_ps(_mm512_srai_epi32(_mm512_add_epi32(dhalf_si512, _mm512_load_si512(ptr + 32)), dshift)));
            _mm512_store_ps(ptr + 48, _mm512_cvtepi32_ps(_mm512_srai_epi32(_mm512_add_epi32(dhalf_si512, _mm512_load_si512(ptr + 48)), dshift)));
            _mm512_store_ps(ptr + 64, _mm512_cvtepi32_ps(_mm512_srai_epi32(_mm512_add_epi32(dhalf_si512, _mm512_load_si512(ptr + 64)), dshift)));
            _mm512_store_ps(ptr + 80, _mm512_cvtepi32_ps(_mm512_srai_epi32(_mm512_add_epi32(dhalf_si512, _mm512_load_si512(ptr + 80)), dshift)));
            _mm512_store_ps(ptr + 96, _mm512_cvtepi32_ps(_mm512_srai_epi32(_mm512_add_epi32(dhalf_si512, _mm512_load_si512(ptr + 96)), dshift)));
            _mm512_store_ps(ptr + 112, _mm512_cvtepi32_ps(_mm512_srai_epi32(_mm512_add_epi32(dhalf_si512, _mm512_load_si512(ptr + 112)), dshift)));
        }
    }

    /////// STEP: compute fvec from coeff ///////
    for (long ind = ESD - 1; ind >= 0; ind--) {
        __m512 y0 = _mm512_mul_ps(_mm512_load_ps(coeff + 2560 * 0 + 0 * 16), lfp_ps16[ESD+0][ind]);
        __m512 y1 = _mm512_mul_ps(_mm512_load_ps(coeff + 2560 * 1 + 0 * 16), lfp_ps16[ESD+0][ind]);
        __m512 y2 = _mm512_mul_ps(_mm512_load_ps(coeff + 2560 * 2 + 0 * 16), lfp_ps16[ESD+0][ind]);
        __m512 y3 = _mm512_mul_ps(_mm512_load_ps(coeff + 2560 * 3 + 0 * 16), lfp_ps16[ESD+0][ind]);
        __m512 y4 = _mm512_mul_ps(_mm512_load_ps(coeff + 2560 * 0 + 1 * 16), lfp_ps16[ESD+1][ind]);
        __m512 y5 = _mm512_mul_ps(_mm512_load_ps(coeff + 2560 * 1 + 1 * 16), lfp_ps16[ESD+1][ind]);
        __m512 y6 = _mm512_mul_ps(_mm512_load_ps(coeff + 2560 * 2 + 1 * 16), lfp_ps16[ESD+1][ind]);
        __m512 y7 = _mm512_mul_ps(_mm512_load_ps(coeff + 2560 * 3 + 1 * 16), lfp_ps16[ESD+1][ind]);

        long i = 2;
        while (i < CSD - 7) {
            y0 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 0 + (i+0) * 16), lfp_ps16[ESD+i+0][ind], y0);
            y1 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 1 + (i+0) * 16), lfp_ps16[ESD+i+0][ind], y1);
            y2 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 2 + (i+0) * 16), lfp_ps16[ESD+i+0][ind], y2);
            y3 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 3 + (i+0) * 16), lfp_ps16[ESD+i+0][ind], y3);
            y4 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 0 + (i+1) * 16), lfp_ps16[ESD+i+1][ind], y4);
            y5 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 1 + (i+1) * 16), lfp_ps16[ESD+i+1][ind], y5);
            y6 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 2 + (i+1) * 16), lfp_ps16[ESD+i+1][ind], y6);
            y7 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 3 + (i+1) * 16), lfp_ps16[ESD+i+1][ind], y7);
            y0 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 0 + (i+2) * 16), lfp_ps16[ESD+i+2][ind], y0);
            y1 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 1 + (i+2) * 16), lfp_ps16[ESD+i+2][ind], y1);
            y2 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 2 + (i+2) * 16), lfp_ps16[ESD+i+2][ind], y2);
            y3 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 3 + (i+2) * 16), lfp_ps16[ESD+i+2][ind], y3);
            y4 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 0 + (i+3) * 16), lfp_ps16[ESD+i+3][ind], y4);
            y5 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 1 + (i+3) * 16), lfp_ps16[ESD+i+3][ind], y5);
            y6 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 2 + (i+3) * 16), lfp_ps16[ESD+i+3][ind], y6);
            y7 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 3 + (i+3) * 16), lfp_ps16[ESD+i+3][ind], y7);
            y0 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 0 + (i+4) * 16), lfp_ps16[ESD+i+4][ind], y0);
            y1 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 1 + (i+4) * 16), lfp_ps16[ESD+i+4][ind], y1);
            y2 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 2 + (i+4) * 16), lfp_ps16[ESD+i+4][ind], y2);
            y3 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 3 + (i+4) * 16), lfp_ps16[ESD+i+4][ind], y3);
            y4 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 0 + (i+5) * 16), lfp_ps16[ESD+i+5][ind], y4);
            y5 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 1 + (i+5) * 16), lfp_ps16[ESD+i+5][ind], y5);
            y6 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 2 + (i+5) * 16), lfp_ps16[ESD+i+5][ind], y6);
            y7 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 3 + (i+5) * 16), lfp_ps16[ESD+i+5][ind], y7);
            y0 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 0 + (i+6) * 16), lfp_ps16[ESD+i+6][ind], y0);
            y1 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 1 + (i+6) * 16), lfp_ps16[ESD+i+6][ind], y1);
            y2 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 2 + (i+6) * 16), lfp_ps16[ESD+i+6][ind], y2);
            y3 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 3 + (i+6) * 16), lfp_ps16[ESD+i+6][ind], y3);
            y4 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 0 + (i+7) * 16), lfp_ps16[ESD+i+7][ind], y4);
            y5 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 1 + (i+7) * 16), lfp_ps16[ESD+i+7][ind], y5);
            y6 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 2 + (i+7) * 16), lfp_ps16[ESD+i+7][ind], y6);
            y7 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 3 + (i+7) * 16), lfp_ps16[ESD+i+7][ind], y7);
            i += 8;
        }
        y0 = _mm512_add_ps(y0, y4);
        y1 = _mm512_add_ps(y1, y5);
        y2 = _mm512_add_ps(y2, y6);
        y3 = _mm512_add_ps(y3, y7);
        while (i < CSD) {
            y0 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 0 + (i+0) * 16), lfp_ps16[ESD+i+0][ind], y0);
            y1 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 1 + (i+0) * 16), lfp_ps16[ESD+i+0][ind], y1);
            y2 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 2 + (i+0) * 16), lfp_ps16[ESD+i+0][ind], y2);
            y3 = _mm512_fmadd_ps(_mm512_load_ps(coeff + 2560 * 3 + (i+0) * 16), lfp_ps16[ESD+i+0][ind], y3);
            i++;
        }
        fvec_ps16[ind][0] = y0;
        fvec_ps16[ind][1] = y1;
        fvec_ps16[ind][2] = y2;
        fvec_ps16[ind][3] = y3;
    }
    
    /////// STEP: size reduce fvec while updating score ///////
    score_ps16[0] = _mm512_cvtepi32_ps(_mm512_load_si512(src_norm));
    score_ps16[1] = _mm512_cvtepi32_ps(_mm512_load_si512(src_norm + 16));
    score_ps16[2] = _mm512_cvtepi32_ps(_mm512_load_si512(src_norm + 32));
    score_ps16[3] = _mm512_cvtepi32_ps(_mm512_load_si512(src_norm + 48));
    norm_ps16[0] = _mm512_mul_ps(iratio_ps16, score_ps16[0]);
    norm_ps16[1] = _mm512_mul_ps(iratio_ps16, score_ps16[1]);
    norm_ps16[2] = _mm512_mul_ps(iratio_ps16, score_ps16[2]);
    norm_ps16[3] = _mm512_mul_ps(iratio_ps16, score_ps16[3]);
    __m512i q0, q1, q2, q3;
    for (long ind = ESD - 1; ind >= 0; ind--) {
        q0 = _mm512_roundscale_ps(_mm512_mul_ps(fvec_ps16[ind][0], inorm_ps16[ind]), _MM_FROUND_TO_NEAREST_INT);
        q1 = _mm512_roundscale_ps(_mm512_mul_ps(fvec_ps16[ind][1], inorm_ps16[ind]), _MM_FROUND_TO_NEAREST_INT);
        q2 = _mm512_roundscale_ps(_mm512_mul_ps(fvec_ps16[ind][2], inorm_ps16[ind]), _MM_FROUND_TO_NEAREST_INT);
        q3 = _mm512_roundscale_ps(_mm512_mul_ps(fvec_ps16[ind][3], inorm_ps16[ind]), _MM_FROUND_TO_NEAREST_INT);
        fvec_ps16[ind][0] = _mm512_fnmadd_ps(q0, lfp_ps16[ind][ind], fvec_ps16[ind][0]);
        fvec_ps16[ind][1] = _mm512_fnmadd_ps(q1, lfp_ps16[ind][ind], fvec_ps16[ind][1]);
        fvec_ps16[ind][2] = _mm512_fnmadd_ps(q2, lfp_ps16[ind][ind], fvec_ps16[ind][2]);
        fvec_ps16[ind][3] = _mm512_fnmadd_ps(q3, lfp_ps16[ind][ind], fvec_ps16[ind][3]);
        norm_ps16[0] = _mm512_fmadd_ps(fvec_ps16[ind][0], fvec_ps16[ind][0], norm_ps16[0]);
        norm_ps16[1] = _mm512_fmadd_ps(fvec_ps16[ind][1], fvec_ps16[ind][1], norm_ps16[1]);
        norm_ps16[2] = _mm512_fmadd_ps(fvec_ps16[ind][2], fvec_ps16[ind][2], norm_ps16[2]);
        norm_ps16[3] = _mm512_fmadd_ps(fvec_ps16[ind][3], fvec_ps16[ind][3], norm_ps16[3]);
        score_ps16[0] = _mm512_min_ps(score_ps16[0], _mm512_mul_ps(norm_ps16[0], ighc_ps16[ind]));
        score_ps16[1] = _mm512_min_ps(score_ps16[1], _mm512_mul_ps(norm_ps16[1], ighc_ps16[ind]));
        score_ps16[2] = _mm512_min_ps(score_ps16[2], _mm512_mul_ps(norm_ps16[2], ighc_ps16[ind]));
        score_ps16[3] = _mm512_min_ps(score_ps16[3], _mm512_mul_ps(norm_ps16[3], ighc_ps16[ind]));

        for (long i = 0; i < ind; i++) {
            fvec_ps16[i][0] = _mm512_fnmadd_ps(q0, lfp_ps16[ind][i], fvec_ps16[i][0]);
            fvec_ps16[i][1] = _mm512_fnmadd_ps(q1, lfp_ps16[ind][i], fvec_ps16[i][1]);
            fvec_ps16[i][2] = _mm512_fnmadd_ps(q2, lfp_ps16[ind][i], fvec_ps16[i][2]);
            fvec_ps16[i][3] = _mm512_fnmadd_ps(q3, lfp_ps16[ind][i], fvec_ps16[i][3]);
        }
    }
    #endif
}


int booster_amx160_t::reconstruct_score(uint16_t *dst, int8_t *src, int32_t *src_norm, long N) {
    const __m512 half_ps16 = _mm512_set1_ps(0.5f);
    #pragma omp parallel for
    for (long thread = 0; thread < p->num_threads; thread++) {
        TILE_INITIALIZE;
        __m512 *score_ps16 = _score_ps16 + thread * 4;
        const long nblocks = (N + 63) / 64;
        long begin_ind = 64 * (nblocks * thread) / p->num_threads;
        long end_ind = 64 * (nblocks * (thread + 1)) / p->num_threads;
        if (end_ind > N) end_ind = N;

        long ind = begin_ind;
        while (ind < end_ind - 63) {
            _process_block64(src + ind * 160, src_norm + ind, thread);
            _mm256_store_si256((__m256i *)(dst + ind), _mm512_cvtepi32_epi16(_mm512_mul_ps(half_ps16, _mm512_cvtps_epi32(score_ps16[0]))));
            _mm256_store_si256((__m256i *)(dst + ind + 16), _mm512_cvtepi32_epi16(_mm512_mul_ps(half_ps16, _mm512_cvtps_epi32(score_ps16[1]))));
            _mm256_store_si256((__m256i *)(dst + ind + 32), _mm512_cvtepi32_epi16(_mm512_mul_ps(half_ps16, _mm512_cvtps_epi32(score_ps16[2]))));
            _mm256_store_si256((__m256i *)(dst + ind + 48), _mm512_cvtepi32_epi16(_mm512_mul_ps(half_ps16, _mm512_cvtps_epi32(score_ps16[3]))));
            ind += 64;
        }

        if (ind < end_ind) {
            __attribute__ ((aligned (64))) int8_t _src[64 * 160];
            __attribute__ ((aligned (64))) int32_t _src_norm[64];
            __attribute__ ((aligned (64))) uint16_t _dst[64];
            for (long i = ind; i < end_ind; i++) {
                _mm512_storeu_si512(_src + (i - ind) * 160, _mm512_loadu_si512(src + i * 160));
                _mm512_storeu_si512(_src + (i - ind) * 160 + 64, _mm512_loadu_si512(src + i * 160 + 64));
                _mm256_store_si256((__m256i *)(_src + (i - ind) * 160 + 128), _mm256_load_si256((__m256i *)(src + i * 160 + 128)));
                _src_norm[i - ind] = src_norm[i];
            }
            _process_block64(_src, _src_norm, thread);
            _mm256_store_si256((__m256i *)_dst, _mm512_cvtepi32_epi16(_mm512_mul_ps(half_ps16, _mm512_cvtps_epi32(score_ps16[0]))));
            _mm256_store_si256((__m256i *)(_dst + 16), _mm512_cvtepi32_epi16(_mm512_mul_ps(half_ps16, _mm512_cvtps_epi32(score_ps16[1]))));
            _mm256_store_si256((__m256i *)(_dst + 32), _mm512_cvtepi32_epi16(_mm512_mul_ps(half_ps16, _mm512_cvtps_epi32(score_ps16[2]))));
            _mm256_store_si256((__m256i *)(_dst + 48), _mm512_cvtepi32_epi16(_mm512_mul_ps(half_ps16, _mm512_cvtps_epi32(score_ps16[3]))));
            for (long i = 0; i < end_ind - ind; i++) {
                dst[ind + i] = _dst[i];
            }
        }
    }
    return 0;
}
int booster_amx160_t::reconstruct_all_score() {
    const __m512 half_ps16 = _mm512_set1_ps(0.5f);
    int8_t *src = p->vec;
    int32_t *src_norm = p->vnorm;
    long N = p->num_vec;

    #pragma omp parallel for
    for (long thread = 0; thread < p->num_threads; thread++) {
        TILE_INITIALIZE;
        __m512 *score_ps16 = _score_ps16 + thread * 4;
        const long nblocks = (N + 63) / 64;
        long begin_ind = 64 * (nblocks * thread) / p->num_threads;
        long end_ind = 64 * (nblocks * (thread + 1)) / p->num_threads;
        if (end_ind > N) end_ind = N;
        fflush(stdout);

        __attribute__ ((aligned (64)))  uint16_t _dst[64];
        long ind = begin_ind;
        while (ind < end_ind - 63) {
            _process_block64(src + ind * 160, src_norm + ind, thread);
            _mm256_store_si256((__m256i *)_dst, _mm512_cvtepi32_epi16(_mm512_mul_ps(half_ps16, _mm512_cvtps_epi32(score_ps16[0]))));
            _mm256_store_si256((__m256i *)(_dst + 16), _mm512_cvtepi32_epi16(_mm512_mul_ps(half_ps16, _mm512_cvtps_epi32(score_ps16[1]))));
            _mm256_store_si256((__m256i *)(_dst + 32), _mm512_cvtepi32_epi16(_mm512_mul_ps(half_ps16, _mm512_cvtps_epi32(score_ps16[2]))));
            _mm256_store_si256((__m256i *)(_dst + 48), _mm512_cvtepi32_epi16(_mm512_mul_ps(half_ps16, _mm512_cvtps_epi32(score_ps16[3]))));
            for (long i = 0; i < 64; i++) {
                p->cvec[3LL * (ind + i) + 2LL] = _dst[i];
            }
            ind += 64;
        }

        if (ind < end_ind) {
            __attribute__ ((aligned (64))) int8_t _src[64 * 160];
            __attribute__ ((aligned (64))) int32_t _src_norm[64];
            for (long i = ind; i < end_ind; i++) {
                _mm512_storeu_si512(_src + (i - ind) * 160, _mm512_loadu_si512(src + i * 160));
                _mm512_storeu_si512(_src + (i - ind) * 160 + 64, _mm512_loadu_si512(src + i * 160 + 64));
                _mm256_store_si256((__m256i *)(_src + (i - ind) * 160 + 128), _mm256_load_si256((__m256i *)(src + i * 160 + 128)));
                _src_norm[i - ind] = src_norm[i];
            }
            _process_block64(_src, _src_norm, thread);
            _mm256_store_si256((__m256i *)_dst, _mm512_cvtepi32_epi16(_mm512_mul_ps(half_ps16, _mm512_cvtps_epi32(score_ps16[0]))));
            _mm256_store_si256((__m256i *)(_dst + 16), _mm512_cvtepi32_epi16(_mm512_mul_ps(half_ps16, _mm512_cvtps_epi32(score_ps16[1]))));
            _mm256_store_si256((__m256i *)(_dst + 32), _mm512_cvtepi32_epi16(_mm512_mul_ps(half_ps16, _mm512_cvtps_epi32(score_ps16[2]))));
            _mm256_store_si256((__m256i *)(_dst + 48), _mm512_cvtepi32_epi16(_mm512_mul_ps(half_ps16, _mm512_cvtps_epi32(score_ps16[3]))));
            for (long i = 0; i < end_ind - ind; i++) {
                p->cvec[3LL * (ind + i) + 2LL] = _dst[i];
            }
        }
    }
    return 0;
}
int booster_amx160_t::report_score_stat(int all_below_maxsc) {
    float msc;
    if (all_below_maxsc) {
        uint16_t msc_epi16 = p->cvec[3ULL * (p->sorted_index > 0 ? p->sorted_index - 1 : 0) + 2ULL];
        for (long i = p->sorted_index; i < p->num_vec; i++) {
            if (p->cvec[3ULL * i + 2ULL] > msc_epi16) msc_epi16 = p->cvec[3ULL * i + 2ULL];
        }
        msc = msc_epi16;
    }

    long score_stat[AMX_BOOST_MAX_DIM+1][512] = {};
    pthread_spinlock_t lock;
    pthread_spin_init(&lock, 1);
    float **dbg_lfp = (float **) NEW_MAT(CSD + ESD, ESD, sizeof(float));
    float igh[AMX_BOOST_MAX_DIM+1] = {};
    float ighc[AMX_BOOST_MAX_DIM] = {};
    float inorm[AMX_BOOST_MAX_DIM];
    float iratio = 2.0 / p->_ratio / p->_ratio;
    for (long i = 0; i <= ESD; i++) {
        igh[i] = 1.0 / p->basis->gh(i + p->index_l - ESD, p->index_r);
    }
    for (long i = 0; i < ESD; i++) {
        ighc[i] = 0.5 * ((float *)(ighc_ps16 + i))[0];
        inorm[i] = ((float *)(inorm_ps16 + i))[0];
    }
    
    do {
        Lattice_QP *L = p->basis->b_loc_QP(p->index_l - ESD, p->index_r);
        Lattice_QP *L_loc_target = new Lattice_QP(CSD, CSD);
        for (long i = 0; i < CSD; i++) {
            for (long j = 0; j < CSD; j++) L_loc_target->get_b().hi[i][j] = p->_b_local[i][j] / p->_ratio;
        }
        L->trans_to(ESD, ESD + CSD, L_loc_target);
        for (long i = ESD; i < ESD + CSD; i++) {
            for (long j = ESD - 1; j >= 0; j--) {
                long q = round(L->get_b().hi[i][j] / L->get_b().hi[j][j]);
                red(L->get_b().hi[i], L->get_b().lo[i], L->get_b().hi[j], L->get_b().lo[j], NTL::quad_float(q), ESD);
            }
        }
        delete L_loc_target;
        for (long i = 0; i < ESD; i++) {
            for (long j = 0; j <= i; j++) {
                dbg_lfp[i][j] = L->get_b().hi[i][j];
            }
        }
        for (long i = ESD; i < ESD + CSD; i++) {
            for (long j = 0; j < ESD; j++) {
                dbg_lfp[i][j] = L->get_b().hi[i][j];
            }
        }
        delete L;
    } while (0);

    #pragma omp parallel for 
    for (long thread = 0; thread < p->num_threads; thread++) {
        long _score_stat[AMX_BOOST_MAX_DIM+1][512] = {};
        long begin_ind = (p->num_vec * thread) / p->num_threads;
        long end_ind = (p->num_vec * (thread + 1)) / p->num_threads;
        long _ind = begin_ind;
        while (_ind < end_ind) {
            float tsc = p->cvec[3LL * _ind + 2LL];
            long ind = ((uint32_t *)(p->cvec + 3LL * _ind))[0];
            __attribute__ ((aligned (64))) int32_t dbg_coeff[160];
            __attribute__ ((aligned (64))) float dbg_fvec[AMX_BOOST_MAX_DIM];
            p->compute_coeff(dbg_coeff, ind);
            set_zero_avx2(dbg_fvec, AMX_BOOST_MAX_DIM);
            for (long j = 0; j < CSD; j++) {
                red_avx2(dbg_fvec, dbg_lfp[ESD+j], -dbg_coeff[j], AMX_BOOST_MAX_DIM);
            }
            float nn = p->vnorm[ind] *iratio;
            float sc = p->vnorm[ind] * 0.5;
            _score_stat[ESD][(int)ceil((100 * sqrt(nn)*igh[ESD]))]++;
            for (long j = ESD - 1; j >= 0; j--) {
                if (!all_below_maxsc && tsc > sc - 5) break;
                float q = round(dbg_fvec[j] * inorm[j]);
                red_avx2(dbg_fvec, dbg_lfp[j], q, AMX_BOOST_MAX_DIM);
                nn += dbg_fvec[j] * dbg_fvec[j];
                sc = sc < nn * ighc[j] ? sc : nn * ighc[j];
                if (!((all_below_maxsc == 1) && nn * ighc[j] > 2.0 * msc)) {
                    _score_stat[j][(int)ceil((100 * sqrt(nn)*igh[j]))]++;
                }
                
            }
            _ind++;
        }
        pthread_spin_lock(&lock);
        for (long i = 0; i < AMX_BOOST_MAX_DIM+1; i++) {
            for (long j = 0; j < 512; j++) {
                score_stat[i][j] += _score_stat[i][j];
            }
        }
        pthread_spin_unlock(&lock);
    }
    for (long i = 0; i < AMX_BOOST_MAX_DIM+1; i++) {
        for (long j = 1; j < 512; j++) {
            score_stat[i][j] += score_stat[i][j-1];
        }
    }
    for (long i = 0; i <= ESD; i++) {
        printf("ESD-%ld: ", ESD - i);
        for (long j = 100; j < 130; j++) {
            printf("%ld(%.2f) ", score_stat[i][j], 2.0 * score_stat[i][j] / pow(j / 100.0, CSD + ESD - i));
        }
        printf("\n");
    }
    FREE_MAT(dbg_lfp);
    return 0;
}
double booster_amx160_t::filter_sol_list(sol_list_amx_t *sol_list, int32_t goal_score) {
    TILE_INITIALIZE;
    long thread = omp_get_thread_num();
    if (thread < 0 || thread >= AMX_MAX_NTHREADS) {
        fprintf(stderr, "[Error] booster_amx160_t::filter_sol_list: thread id (%ld) out of range\n", thread);
    }
    __m512 *score_ps16 = _score_ps16 + thread * 4;
    const __m512 th_ps16 = _mm512_set1_ps(2 * goal_score);
    const __m512i all0x80 = _mm512_set1_epi8(0x80);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    long pind = sol_list->num_a_insert;
    long nind = sol_list->num_s_insert;
    long end_pind = sol_list->num_a;
    long end_nind = sol_list->num_s;
    __attribute__ ((aligned (64))) int8_t buf[64 * 160];
    __attribute__ ((aligned (64))) int32_t buf_norm[64];
    while (pind < end_pind - 63) {
        for (long ind = 0; ind < 64; ind += 8) {
            COMPUTE_VEC_AND_NORM_B8((buf + ind * 160), (buf_norm + ind), sol_list->a_list + (pind + ind) * 2, 1);
        }
        _process_block64(buf, buf_norm, thread);
        __mmask16 msk0 = _mm512_cmp_ps_mask(score_ps16[0], th_ps16, _CMP_LT_OQ);
        __mmask16 msk1 = _mm512_cmp_ps_mask(score_ps16[1], th_ps16, _CMP_LT_OQ);
        __mmask16 msk2 = _mm512_cmp_ps_mask(score_ps16[2], th_ps16, _CMP_LT_OQ);
        __mmask16 msk3 = _mm512_cmp_ps_mask(score_ps16[3], th_ps16, _CMP_LT_OQ);
        uint64_t msk = (((uint64_t)msk0)) | (((uint64_t)msk1) << 16) | (((uint64_t)msk2) << 32) | (((uint64_t)msk3) << 48);
        while (msk) {
            long r = __builtin_ctzll(msk);
            msk &= ~(1ULL << r);
            uint64_t u = p->vu[sol_list->a_list[(pind + r) * 2]] + p->vu[sol_list->a_list[(pind + r) * 2 + 1]];
            if (p->uid->insert_uid(u)) {
                ((uint64_t *)(sol_list->a_list))[sol_list->num_a_insert] = ((uint64_t *)(sol_list->a_list))[pind + r];
                sol_list->num_a_insert++;
            }
        }
        pind += 64;
    }

    if (pind < end_pind) {
        long ind = 0;
        for (; ind < end_pind - pind - 7; ind += 8) {
            COMPUTE_VEC_AND_NORM_B8((buf + ind * 160), (buf_norm + ind), sol_list->a_list + (pind + ind) * 2, 1);
        }
        for (; ind < end_pind - pind; ind++) {
            COMPUTE_VEC_AND_NORM((buf + ind * 160), (buf_norm + ind), sol_list->a_list + (pind + ind) * 2, 1);
        }
        _process_block64(buf, buf_norm, thread);
        __mmask16 msk0 = _mm512_cmp_ps_mask(score_ps16[0], th_ps16, _CMP_LT_OQ);
        __mmask16 msk1 = _mm512_cmp_ps_mask(score_ps16[1], th_ps16, _CMP_LT_OQ);
        __mmask16 msk2 = _mm512_cmp_ps_mask(score_ps16[2], th_ps16, _CMP_LT_OQ);
        __mmask16 msk3 = _mm512_cmp_ps_mask(score_ps16[3], th_ps16, _CMP_LT_OQ);
        uint64_t msk = (((uint64_t)msk0)) | (((uint64_t)msk1) << 16) | (((uint64_t)msk2) << 32) | (((uint64_t)msk3) << 48);
        msk &= (1 << (end_pind - pind)) - 1;
        while (msk) {
            long r = __builtin_ctzll(msk);
            msk &= ~(1ULL << r);
            uint64_t u = p->vu[sol_list->a_list[(pind + r) * 2]] + p->vu[sol_list->a_list[(pind + r) * 2 + 1]];
            if (p->uid->insert_uid(u)) {
                ((uint64_t *)(sol_list->a_list))[sol_list->num_a_insert] = ((uint64_t *)(sol_list->a_list))[pind + r];
                sol_list->num_a_insert++;
            }
        }
    }
    
    sol_list->num_a = sol_list->num_a_insert;

    while (nind < end_nind - 63) {
        for (long ind = 0; ind < 64; ind += 8) {
            COMPUTE_VEC_AND_NORM_B8((buf + ind * 160), (buf_norm + ind), sol_list->s_list + (nind + ind) * 2, 0);
        }
        _process_block64(buf, buf_norm, thread);
        __mmask16 msk0 = _mm512_cmp_ps_mask(score_ps16[0], th_ps16, _CMP_LT_OQ);
        __mmask16 msk1 = _mm512_cmp_ps_mask(score_ps16[1], th_ps16, _CMP_LT_OQ);
        __mmask16 msk2 = _mm512_cmp_ps_mask(score_ps16[2], th_ps16, _CMP_LT_OQ);
        __mmask16 msk3 = _mm512_cmp_ps_mask(score_ps16[3], th_ps16, _CMP_LT_OQ);
        uint64_t msk = (((uint64_t)msk0)) | (((uint64_t)msk1) << 16) | (((uint64_t)msk2) << 32) | (((uint64_t)msk3) << 48);
        while (msk) {
            long r = __builtin_ctzll(msk);
            msk &= ~(1ULL << r);
            uint64_t u = p->vu[sol_list->s_list[(nind + r) * 2]] - p->vu[sol_list->s_list[(nind + r) * 2 + 1]];
            if (p->uid->insert_uid(u)) {
                ((uint64_t *)(sol_list->s_list))[sol_list->num_s_insert] = ((uint64_t *)(sol_list->s_list))[nind + r];
                sol_list->num_s_insert++;
            }
        }
        nind += 64;
    }

    if (nind < end_nind) {
        long ind = 0;
        for (; ind < end_nind - nind - 7; ind += 8) {
            COMPUTE_VEC_AND_NORM_B8((buf + ind * 160), (buf_norm + ind), sol_list->s_list + (nind + ind) * 2, 0);
        }
        for (; ind < end_nind - nind; ind++) {
            COMPUTE_VEC_AND_NORM((buf + ind * 160), (buf_norm + ind), sol_list->s_list + (nind + ind) * 2, 0);
        }
        _process_block64(buf, buf_norm, thread);
        __mmask16 msk0 = _mm512_cmp_ps_mask(score_ps16[0], th_ps16, _CMP_LT_OQ);
        __mmask16 msk1 = _mm512_cmp_ps_mask(score_ps16[1], th_ps16, _CMP_LT_OQ);
        __mmask16 msk2 = _mm512_cmp_ps_mask(score_ps16[2], th_ps16, _CMP_LT_OQ);
        __mmask16 msk3 = _mm512_cmp_ps_mask(score_ps16[3], th_ps16, _CMP_LT_OQ);
        uint64_t msk = (((uint64_t)msk0)) | (((uint64_t)msk1) << 16) | (((uint64_t)msk2) << 32) | (((uint64_t)msk3) << 48);
        msk &= (1 << (end_nind - nind)) - 1;
        while (msk) {
            long r = __builtin_ctzll(msk);
            msk &= ~(1ULL << r);
            uint64_t u = p->vu[sol_list->s_list[(nind + r) * 2]] - p->vu[sol_list->s_list[(nind + r) * 2 + 1]];
            if (p->uid->insert_uid(u)) {
                ((uint64_t *)(sol_list->s_list))[sol_list->num_s_insert] = ((uint64_t *)(sol_list->s_list))[nind + r];
                sol_list->num_s_insert++;
            }
        }
    }

    sol_list->num_s = sol_list->num_s_insert;

    gettimeofday(&end, NULL);
    sol_list->filter_time += (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
}
#endif