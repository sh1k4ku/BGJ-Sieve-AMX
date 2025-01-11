#ifndef __BGJ_AMX_H
#define __BGJ_AMX_H

///////////////// BGJ3 parameters /////////////////
#define BGJ3_AMX_BUCKET0_ALPHA      0.210
#define BGJ3_AMX_BUCKET1_ALPHA      0.215
#define BGJ3_AMX_BUCKET2_ALPHA      0.285
#define BGJ3_AMX_REUSE0_ALPHA       0.375
#define BGJ3_AMX_REUSE1_ALPHA       0.310
#define BGJ3_AMX_BUCKET0_BATCHSIZE    128
#define BGJ3_AMX_BUCKET1_BATCHSIZE     64
#define BGJ3_AMX_BUCKET2_BATCHSIZE    256
#define BGJ3_AMX_USE_FARAWAY_CENTER     1


#define AMX_MAX_NTHREADS 112
#define AMX_MIN_LOG_CSD 45
#define AMX_MAX_STUCK_TIME 2
#define AMX_BUCKET2_USE_BUFFER 0
#define AMX_PARALLEL_BUCKET1 0
#define AMX_MAX_NUM_TRYCENTER 16384
#define AMX_BOOST_MAX_DIM 48
#define AMX_BOOST_PREFER_DEEP 1.00
#define AMX_BOOST_GOAL_NORM_SCALE 1.14
#define AMX_BOOST_SATURATION_RATIO 0.375
#define AMX_BOOST_DOWNSIEVE_MASK_DIM 8
#define AMX_BOOST_DOWNSIEVE_MASK_RATIO 0.005

#define _CEIL16(__x) (((__x) + 15ULL) & ~15ULL)

struct __tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16]; 
    uint8_t rows[16]; 
};

#define TILE_INITIALIZE do {        \
    __tile_config tile_config = {}; \
    tile_config.palette_id = 1;     \
    tile_config.start_row = 0;      \
                                    \
    for (long i = 0; i < 8; i++) {  \
        tile_config.rows[i] = 16;   \
        tile_config.colsb[i] = 64;  \
    }                               \
    _tile_loadconfig(&tile_config); \
} while (0)

#define AVX512_MATTR_16x16(__dst, __dst_tr, __z0, __z1, __z2, __z3, __z4, __z5, __z6, __z7, __z8, __z9, __za, __zb, __zc, __zd, __ze, __zf) \
                                                                                                                                    do { \
    __m512i __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7, __t8, __t9, __ta, __tb, __tc, __td, __te, __tf; \
    if (__dst) _mm512_store_si512((__m512i *)(__dst) + 0, __z0);   \
    if (__dst) _mm512_store_si512((__m512i *)(__dst) + 1, __z1);   \
    if (__dst) _mm512_store_si512((__m512i *)(__dst) + 2, __z2);   \
    if (__dst) _mm512_store_si512((__m512i *)(__dst) + 3, __z3);   \
    if (__dst) _mm512_store_si512((__m512i *)(__dst) + 4, __z4);   \
    if (__dst) _mm512_store_si512((__m512i *)(__dst) + 5, __z5);   \
    if (__dst) _mm512_store_si512((__m512i *)(__dst) + 6, __z6);   \
    if (__dst) _mm512_store_si512((__m512i *)(__dst) + 7, __z7);   \
    if (__dst) _mm512_store_si512((__m512i *)(__dst) + 8, __z8);   \
    if (__dst) _mm512_store_si512((__m512i *)(__dst) + 9, __z9);   \
    if (__dst) _mm512_store_si512((__m512i *)(__dst) + 10, __za);  \
    if (__dst) _mm512_store_si512((__m512i *)(__dst) + 11, __zb);  \
    if (__dst) _mm512_store_si512((__m512i *)(__dst) + 12, __zc);  \
    if (__dst) _mm512_store_si512((__m512i *)(__dst) + 13, __zd);  \
    if (__dst) _mm512_store_si512((__m512i *)(__dst) + 14, __ze);  \
    if (__dst) _mm512_store_si512((__m512i *)(__dst) + 15, __zf);  \
                                                        \
                                                \
    __t0 = _mm512_unpacklo_epi32(__z0,__z1);    \
    __t1 = _mm512_unpackhi_epi32(__z0,__z1);    \
    __t2 = _mm512_unpacklo_epi32(__z2,__z3);    \
    __t3 = _mm512_unpackhi_epi32(__z2,__z3);    \
    __t4 = _mm512_unpacklo_epi32(__z4,__z5);    \
    __t5 = _mm512_unpackhi_epi32(__z4,__z5);    \
    __t6 = _mm512_unpacklo_epi32(__z6,__z7);    \
    __t7 = _mm512_unpackhi_epi32(__z6,__z7);    \
    __t8 = _mm512_unpacklo_epi32(__z8,__z9);    \
    __t9 = _mm512_unpackhi_epi32(__z8,__z9);    \
    __ta = _mm512_unpacklo_epi32(__za,__zb);    \
    __tb = _mm512_unpackhi_epi32(__za,__zb);    \
    __tc = _mm512_unpacklo_epi32(__zc,__zd);    \
    __td = _mm512_unpackhi_epi32(__zc,__zd);    \
    __te = _mm512_unpacklo_epi32(__ze,__zf);    \
    __tf = _mm512_unpackhi_epi32(__ze,__zf);    \
                                                \
    __z0 = _mm512_unpacklo_epi64(__t0,__t2);    \
    __z1 = _mm512_unpackhi_epi64(__t0,__t2);    \
    __z2 = _mm512_unpacklo_epi64(__t1,__t3);    \
    __z3 = _mm512_unpackhi_epi64(__t1,__t3);    \
    __z4 = _mm512_unpacklo_epi64(__t4,__t6);    \
    __z5 = _mm512_unpackhi_epi64(__t4,__t6);    \
    __z6 = _mm512_unpacklo_epi64(__t5,__t7);    \
    __z7 = _mm512_unpackhi_epi64(__t5,__t7);    \
    __z8 = _mm512_unpacklo_epi64(__t8,__ta);    \
    __z9 = _mm512_unpackhi_epi64(__t8,__ta);    \
    __za = _mm512_unpacklo_epi64(__t9,__tb);    \
    __zb = _mm512_unpackhi_epi64(__t9,__tb);    \
    __zc = _mm512_unpacklo_epi64(__tc,__te);    \
    __zd = _mm512_unpackhi_epi64(__tc,__te);    \
    __ze = _mm512_unpacklo_epi64(__td,__tf);    \
    __zf = _mm512_unpackhi_epi64(__td,__tf);    \
                                                \
    __t0 = _mm512_shuffle_i64x2(__z0, __z4, 0x88);\
    __t1 = _mm512_shuffle_i64x2(__z1, __z5, 0x88);\
    __t2 = _mm512_shuffle_i64x2(__z2, __z6, 0x88);\
    __t3 = _mm512_shuffle_i64x2(__z3, __z7, 0x88);\
    __t4 = _mm512_shuffle_i64x2(__z0, __z4, 0xdd);\
    __t5 = _mm512_shuffle_i64x2(__z1, __z5, 0xdd);\
    __t6 = _mm512_shuffle_i64x2(__z2, __z6, 0xdd);\
    __t7 = _mm512_shuffle_i64x2(__z3, __z7, 0xdd);\
    __t8 = _mm512_shuffle_i64x2(__z8, __zc, 0x88);\
    __t9 = _mm512_shuffle_i64x2(__z9, __zd, 0x88);\
    __ta = _mm512_shuffle_i64x2(__za, __ze, 0x88);\
    __tb = _mm512_shuffle_i64x2(__zb, __zf, 0x88);\
    __tc = _mm512_shuffle_i64x2(__z8, __zc, 0xdd);\
    __td = _mm512_shuffle_i64x2(__z9, __zd, 0xdd);\
    __te = _mm512_shuffle_i64x2(__za, __ze, 0xdd);\
    __tf = _mm512_shuffle_i64x2(__zb, __zf, 0xdd);\
                                                  \
    __z0 = _mm512_shuffle_i64x2(__t0, __t8, 0x88);\
    __z1 = _mm512_shuffle_i64x2(__t1, __t9, 0x88);\
    __z2 = _mm512_shuffle_i64x2(__t2, __ta, 0x88);\
    __z3 = _mm512_shuffle_i64x2(__t3, __tb, 0x88);\
    __z4 = _mm512_shuffle_i64x2(__t4, __tc, 0x88);\
    __z5 = _mm512_shuffle_i64x2(__t5, __td, 0x88);\
    __z6 = _mm512_shuffle_i64x2(__t6, __te, 0x88);\
    __z7 = _mm512_shuffle_i64x2(__t7, __tf, 0x88);\
    __z8 = _mm512_shuffle_i64x2(__t0, __t8, 0xdd);\
    __z9 = _mm512_shuffle_i64x2(__t1, __t9, 0xdd);\
    __za = _mm512_shuffle_i64x2(__t2, __ta, 0xdd);\
    __zb = _mm512_shuffle_i64x2(__t3, __tb, 0xdd);\
    __zc = _mm512_shuffle_i64x2(__t4, __tc, 0xdd);\
    __zd = _mm512_shuffle_i64x2(__t5, __td, 0xdd);\
    __ze = _mm512_shuffle_i64x2(__t6, __te, 0xdd);\
    __zf = _mm512_shuffle_i64x2(__t7, __tf, 0xdd);\
                                                  \
    _mm512_store_si512((__m512i *)(__dst_tr) + 0, __z0);   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 1, __z1);   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 2, __z2);   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 3, __z3);   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 4, __z4);   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 5, __z5);   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 6, __z6);   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 7, __z7);   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 8, __z8);   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 9, __z9);   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 10, __za);  \
    _mm512_store_si512((__m512i *)(__dst_tr) + 11, __zb);  \
    _mm512_store_si512((__m512i *)(__dst_tr) + 12, __zc);  \
    _mm512_store_si512((__m512i *)(__dst_tr) + 13, __zd);  \
    _mm512_store_si512((__m512i *)(__dst_tr) + 14, __ze);  \
    _mm512_store_si512((__m512i *)(__dst_tr) + 15, __zf);  \
} while (0)

#define AVX512_MATTR_16x8(__dst, __dst_tr, __y0, __y1, __y2, __y3, __y4, __y5, __y6, __y7, __y8, __y9, __ya, __yb, __yc, __yd, __ye, __yf) \
                                                                                                                                    do { \
    __m256i __s0, __s1, __s2, __s3, __s4, __s5, __s6, __s7, __s8, __s9, __sa, __sb, __sc, __sd, __se, __sf; \
    __m512i __z0, __z1, __z2, __z3, __z4, __z5, __z6, __z7; \
    __m512i __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7; \
                                                        \
    if (__dst) _mm256_store_si256((__m256i *)(__dst) + 0, __y0);   \
    if (__dst) _mm256_store_si256((__m256i *)(__dst) + 1, __y1);   \
    if (__dst) _mm256_store_si256((__m256i *)(__dst) + 2, __y2);   \
    if (__dst) _mm256_store_si256((__m256i *)(__dst) + 3, __y3);   \
    if (__dst) _mm256_store_si256((__m256i *)(__dst) + 4, __y4);   \
    if (__dst) _mm256_store_si256((__m256i *)(__dst) + 5, __y5);   \
    if (__dst) _mm256_store_si256((__m256i *)(__dst) + 6, __y6);   \
    if (__dst) _mm256_store_si256((__m256i *)(__dst) + 7, __y7);   \
    if (__dst) _mm256_store_si256((__m256i *)(__dst) + 8, __y8);   \
    if (__dst) _mm256_store_si256((__m256i *)(__dst) + 9, __y9);   \
    if (__dst) _mm256_store_si256((__m256i *)(__dst) + 10, __ya);  \
    if (__dst) _mm256_store_si256((__m256i *)(__dst) + 11, __yb);  \
    if (__dst) _mm256_store_si256((__m256i *)(__dst) + 12, __yc);  \
    if (__dst) _mm256_store_si256((__m256i *)(__dst) + 13, __yd);  \
    if (__dst) _mm256_store_si256((__m256i *)(__dst) + 14, __ye);  \
    if (__dst) _mm256_store_si256((__m256i *)(__dst) + 15, __yf);  \
                                                        \
    __s0 = _mm256_unpacklo_epi32(__y0,__y1);            \
    __s1 = _mm256_unpackhi_epi32(__y0,__y1);            \
    __s2 = _mm256_unpacklo_epi32(__y2,__y3);            \
    __s3 = _mm256_unpackhi_epi32(__y2,__y3);            \
    __s4 = _mm256_unpacklo_epi32(__y4,__y5);            \
    __s5 = _mm256_unpackhi_epi32(__y4,__y5);            \
    __s6 = _mm256_unpacklo_epi32(__y6,__y7);            \
    __s7 = _mm256_unpackhi_epi32(__y6,__y7);            \
    __s8 = _mm256_unpacklo_epi32(__y8,__y9);            \
    __s9 = _mm256_unpackhi_epi32(__y8,__y9);            \
    __sa = _mm256_unpacklo_epi32(__ya,__yb);            \
    __sb = _mm256_unpackhi_epi32(__ya,__yb);            \
    __sc = _mm256_unpacklo_epi32(__yc,__yd);            \
    __sd = _mm256_unpackhi_epi32(__yc,__yd);            \
    __se = _mm256_unpacklo_epi32(__ye,__yf);            \
    __sf = _mm256_unpackhi_epi32(__ye,__yf);            \
                                                        \
    __y0 = _mm256_unpacklo_epi64(__s0,__s2);            \
    __y1 = _mm256_unpackhi_epi64(__s0,__s2);            \
    __y2 = _mm256_unpacklo_epi64(__s1,__s3);            \
    __y3 = _mm256_unpackhi_epi64(__s1,__s3);            \
    __y4 = _mm256_unpacklo_epi64(__s4,__s6);            \
    __y5 = _mm256_unpackhi_epi64(__s4,__s6);            \
    __y6 = _mm256_unpacklo_epi64(__s5,__s7);            \
    __y7 = _mm256_unpackhi_epi64(__s5,__s7);            \
    __y8 = _mm256_unpacklo_epi64(__s8,__sa);            \
    __y9 = _mm256_unpackhi_epi64(__s8,__sa);            \
    __ya = _mm256_unpacklo_epi64(__s9,__sb);            \
    __yb = _mm256_unpackhi_epi64(__s9,__sb);            \
    __yc = _mm256_unpacklo_epi64(__sc,__se);            \
    __yd = _mm256_unpackhi_epi64(__sc,__se);            \
    __ye = _mm256_unpacklo_epi64(__sd,__sf);            \
    __yf = _mm256_unpackhi_epi64(__sd,__sf);            \
                                                        \
    __z0 = _mm512_inserti64x4(_mm512_castsi256_si512(__y0), __y4, 1); \
    __z1 = _mm512_inserti64x4(_mm512_castsi256_si512(__y1), __y5, 1); \
    __z2 = _mm512_inserti64x4(_mm512_castsi256_si512(__y2), __y6, 1); \
    __z3 = _mm512_inserti64x4(_mm512_castsi256_si512(__y3), __y7, 1); \
    __z4 = _mm512_inserti64x4(_mm512_castsi256_si512(__y8), __yc, 1); \
    __z5 = _mm512_inserti64x4(_mm512_castsi256_si512(__y9), __yd, 1); \
    __z6 = _mm512_inserti64x4(_mm512_castsi256_si512(__ya), __ye, 1); \
    __z7 = _mm512_inserti64x4(_mm512_castsi256_si512(__yb), __yf, 1); \
                                                                      \
    __t0 = _mm512_shuffle_i64x2(__z0, __z4, 0x88); \
    __t1 = _mm512_shuffle_i64x2(__z1, __z5, 0x88); \
    __t2 = _mm512_shuffle_i64x2(__z2, __z6, 0x88); \
    __t3 = _mm512_shuffle_i64x2(__z3, __z7, 0x88); \
    __t4 = _mm512_shuffle_i64x2(__z0, __z4, 0xdd); \
    __t5 = _mm512_shuffle_i64x2(__z1, __z5, 0xdd); \
    __t6 = _mm512_shuffle_i64x2(__z2, __z6, 0xdd); \
    __t7 = _mm512_shuffle_i64x2(__z3, __z7, 0xdd); \
                                                   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 0, __t0);   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 1, __t1);   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 2, __t2);   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 3, __t3);   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 4, __t4);   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 5, __t5);   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 6, __t6);   \
    _mm512_store_si512((__m512i *)(__dst_tr) + 7, __t7);   \
} while (0)

#include <atomic>

template <unsigned nb>
struct bgj_amx_profile_data_t {
    Pool_epi8_t<nb> *p;
    pthread_spinlock_t profile_lock;
    struct timeval bgj_start_time, bgj_end_time;

    void init(Pool_epi8_t<nb> *_p, long _log_level);
    void initial_log(int bgj);
    void final_log(int bgj, long sieving_stucked);
    void epoch_initial_log(int32_t goal_norm);
    void one_epoch_log(int bgj);
    void insert_log(uint64_t num_total_sol, double insert_time);
    void insert_inner_log(uint64_t *length_stat, uint64_t num_linfty_failed, uint64_t num_l2_failed, uint64_t num_not_try);
    void pool_bucket_check(bucket_amx_t **bucket_list, long num_bucket, double alpha);
    void subbucket_check(bucket_amx_t **bucket_list, bucket_amx_t **subbucket_list, long num_bucket, long num_subbucket, double alpha);
    void sol_check(sol_list_amx_t **sol_list, long num, int32_t goal_norm, UidHashTable *uid);
    void report_bucket_not_used(int bgj, long nrem0, long nrem1 = 0, long nrem2 = 0);
    void combine(bgj_amx_profile_data_t<nb> *prof);
    

    long log_level = -1;
    FILE *log_out = stdout;
    FILE *log_err = stderr;

    long num_epoch = 0;
    long num_bucket0 = 0;
    long num_bucket1 = 0;
    long num_bucket2 = 0;
    long num_r0 = 0;
    long num_r1 = 0;
    long sum_bucket0_size = 0;
    long sum_bucket1_size = 0;
    long sum_bucket2_size = 0;
    long sum_r0_size = 0;
    long sum_r1_size = 0;

    double bucket0_time = 0.0;
    double bucket1_time = 0.0;
    double bucket2_time = 0.0;
    double search0_time = 0.0;
    double search1_time = 0.0;
    double search2_time = 0.0;
    double sort_time = 0.0;
    double insert_time = 0.0;
    #if BOOST_AMX_SIEVE
    double filter_time = 0.0;
    #endif

    uint64_t bucket0_ndp = 0;
    uint64_t bucket1_ndp = 0;
    uint64_t bucket2_ndp = 0;
    uint64_t search0_ndp = 0;
    uint64_t search1_ndp = 0;
    uint64_t search2_ndp = 0;

    std::atomic<uint64_t> try_add2{0};
    std::atomic<uint64_t> succ_add2{0};
    uint64_t succ_insert = 0;
};

#if BOOST_AMX_SIEVE

struct booster_amx160_t {
    // read only shared varibles
    Pool_epi8_t<5> *p = NULL;
    long CSD, ESD;
    __m512 ighc_ps16[AMX_BOOST_MAX_DIM];
    __m512 inorm_ps16[AMX_BOOST_MAX_DIM];
    __m512 iratio_ps16;
    __m512i dhalf_si512;
    int32_t dshift;
    __attribute__ ((aligned (64))) int8_t b_dual_0[10*1024] = {};
    __attribute__ ((aligned (64))) int8_t b_dual_1[10*1024] = {};
    __attribute__ ((aligned (64))) int8_t b_dual_2[10*1024] = {};
    __m512 **lfp_ps16 = NULL;

    // private varibles
    float *_coeff = NULL;
    __m512 *_score_ps16 = NULL;
    __m512 *_norm_ps16 = NULL;
    __m512 **_fvec_ps16 = NULL;

    booster_amx160_t() {}
    ~booster_amx160_t();

    int init(Pool_epi8_t<5> *_p, long _esd = -1, double _prefer_deep = -1.0);
    int reconstruct_all_score();
    int reconstruct_score(uint16_t *dst, int8_t *src, int32_t *src_norm, long N);
    int report_score_stat(int all_below_maxscore = 0);
    // after calling this function, sol_list->num_a should equal to sol_list->num_a_insert
    double filter_sol_list(sol_list_amx_t *sol_list, int32_t goal_score);
    void _process_block64(int8_t *src, int32_t *src_norm, long thread);


};
#endif
#endif