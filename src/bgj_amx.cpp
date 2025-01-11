#include "../include/config.h"

#if defined(__AMX_INT8__)

#include "../include/pool_epi8.h"
#include "../include/bucket_amx.h"
#include "../include/bgj_amx.h"

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sys/syscall.h>

#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILEDATA      18

#if 1
#include <sys/time.h>
struct timeval _amx_timer_start[AMX_MAX_NTHREADS], _amx_timer_end[AMX_MAX_NTHREADS];
double _amx_time_curr[AMX_MAX_NTHREADS];

#define TIMER_START do {                                                        \
        gettimeofday(&_amx_timer_start[omp_get_thread_num()], NULL);                                  \
    } while (0)

#define TIMER_END do {                                                          \
        gettimeofday(&_amx_timer_end[omp_get_thread_num()], NULL);                                    \
        _amx_time_curr[omp_get_thread_num()] =                                                            \
            (_amx_timer_end[omp_get_thread_num()].tv_sec-_amx_timer_start[omp_get_thread_num()].tv_sec)+                    \
            (double)(_amx_timer_end[omp_get_thread_num()].tv_usec-_amx_timer_start[omp_get_thread_num()].tv_usec)/1000000.0;\
    } while (0)

#define CURRENT_TIME (_amx_time_curr[omp_get_thread_num()])
#endif

///////////////// bgj_profile_data_t impl /////////////////

template <uint32_t nb>
void bgj_amx_profile_data_t<nb>::init(Pool_epi8_t<nb> *_p, long _log_level) {
    p = _p;
    log_level = _log_level;
    pthread_spin_init(&profile_lock, PTHREAD_PROCESS_SHARED);
    gettimeofday(&bgj_start_time, NULL);
}

template <uint32_t nb>
void bgj_amx_profile_data_t<nb>::initial_log(int bgj) {
    if (log_level == 0 && p->CSD > AMX_MIN_LOG_CSD){
    fprintf(log_err, "begin bgj%d-amx sieve on context [%ld, %ld], gh = %.2f, pool size = %ld, %ld threads will be used",
                    bgj, p->index_l, p->index_r, sqrt(p->gh2), p->num_vec, p->num_threads);
    }
    if (log_level >= 1 && p->CSD > AMX_MIN_LOG_CSD) {
        fprintf(log_out, "begin bgj%d sieve, sieving dimension = %ld, pool size = %ld\n", bgj, p->CSD, p->num_vec);
        if (log_level >= 3 && p->CSD > AMX_MIN_LOG_CSD){
            if (bgj == 3) {
                fprintf(log_out, "bucket0_batchsize = %d\n", BGJ3_AMX_BUCKET0_BATCHSIZE);
                fprintf(log_out, "bucket1_batchsize = %d\n", BGJ3_AMX_BUCKET1_BATCHSIZE);
                fprintf(log_out, "bucket2_batchsize = %d\n", BGJ3_AMX_BUCKET2_BATCHSIZE);
                fprintf(log_out, "bucket0_alpha = %f\n", BGJ3_AMX_BUCKET0_ALPHA);
                fprintf(log_out, "bucket1_alpha = %f\n", BGJ3_AMX_BUCKET1_ALPHA);
                fprintf(log_out, "bucket2_alpha = %f\n", BGJ3_AMX_BUCKET2_ALPHA);
                fprintf(log_out, "bucket0_reuse0_alpha = %f\n", BGJ3_AMX_REUSE0_ALPHA);
                fprintf(log_out, "bucket1_reuse1_alpha = %f\n", BGJ3_AMX_REUSE1_ALPHA);
            }
        }
        fflush(log_out);
    }
}

template <uint32_t nb>
void bgj_amx_profile_data_t<nb>::final_log(int bgj, long sieving_stucked) {
    if (log_level >= 1 && p->CSD > AMX_MIN_LOG_CSD) {
        if (sieving_stucked){
            fprintf(log_out, "sieving stucked, aborted.\n");
        } else {
            fprintf(log_out, "sieving done.\n");
        }
        if (bgj == 3) {
            #if BOOST_AMX_SIEVE
            char ftime_str[256];
            char fspeed_str[256];
            sprintf(ftime_str, ", filter: %.3fs", filter_time);
            sprintf(fspeed_str, ", f: %.3f MOPS", succ_add2 / (filter_time + 0.000001) / 1048576.0);
            #endif
            double speed0, speed1, speed2, speedr0, speedr1;
            fprintf(log_out, "solution collect done, found %ld solutions in %ld buckets, %ld(%.3f) inserted, "
                            "b0: %.3fs, b1: %.3fs, b2: %.3fs, s0: %.3fs, s1: %.3fs, s2: %.3fs, sort: %.3fs, insert: %.3fs%s\n",
                     (uint64_t)succ_add2, num_bucket0 + num_bucket1 + num_bucket2 + num_r0 + num_r1, succ_insert, succ_insert / (succ_add2 + 0.000001), 
                     bucket0_time, bucket1_time, bucket2_time, search0_time, search1_time, search2_time, 
                     sort_time, insert_time, BOOST_AMX_SIEVE ? ftime_str : "");
            speed0 = p->CSD * 2 * bucket0_ndp/bucket0_time/1073741824.0/1024.0;
            speed1 = p->CSD * 2 * bucket1_ndp/bucket1_time/1073741824.0/1024.0;
            speed2 = p->CSD * 2 * bucket2_ndp/bucket2_time/1073741824.0/1024.0;
            fprintf(log_out, "b0: %.2f bucket/s (%.3f TFLOPS), b1: %.2f bucket/s (%.3f TFLOPS), b2: %.2f bucket/s (%.3f TFLOPS)\n",
                            num_bucket0/bucket0_time, speed0, num_bucket1/bucket1_time, speed1, num_bucket2/bucket2_time, speed2);
            speed0 = p->CSD * 2 * search0_ndp/search0_time/1073741824.0/1024.0;
            speed1 = p->CSD * 2 * search1_ndp/search1_time/1073741824.0/1024.0;
            speed2 = p->CSD * 2 * search2_ndp/search2_time/1073741824.0/1024.0;
            fprintf(log_out, "s0: %.2f bucket/s (%.3f TFLOPS), s1: %.2f bucket/s (%.3f TFLOPS), "
                            "s2: %.2f bucket/s (%.3f TFLOPS)%s\n",
                            num_bucket0/search0_time, speed0, num_bucket1/search1_time, speed1, 
                            num_bucket2/search2_time, speed2, BOOST_AMX_SIEVE ? fspeed_str : "");
            fprintf(log_out, "nb0 = %ld(%ld), nb1 = %ld(%ld), nb2 = %ld(%ld), nbr0 = %ld(%ld), nbr1 = %ld(%ld)\n", 
                                num_bucket0, (long)(sum_bucket0_size/(0.000001+num_bucket0)), 
                                num_bucket1, (long)(sum_bucket1_size/(0.000001+num_bucket1)), 
                                num_bucket2, (long)(sum_bucket2_size/(0.000001+num_bucket2)),
                                num_r0, (long)(sum_r0_size/(0.000001+num_r0)),
                                num_r1, (long)(sum_r1_size/(0.000001+num_r1)));
            fprintf(log_out, "bucket cost = %ld dp/sol, search cost = %ld dp/sol\n",
                    (long)((bucket0_ndp+bucket1_ndp+bucket2_ndp) / (succ_add2 + 0.000001)),
                    (long)((search0_ndp+search1_ndp+search2_ndp) / (succ_add2 + 0.000001)));
            fprintf(log_out, "try_add2 = %ld, succ_add2 = %ld(1/%.2f), %ld(1/%.2f) inserted\n\n\n",
                    (uint64_t)try_add2, (uint64_t)succ_add2, (try_add2+1e-20)/succ_add2, succ_insert, (succ_add2+1e-20)/succ_insert);
        }
        fflush(log_out);
    }
    if (log_level == 0 && p->CSD > AMX_MIN_LOG_CSD){
        gettimeofday(&bgj_end_time, NULL);
        double tt = bgj_end_time.tv_sec-bgj_start_time.tv_sec+ (double)(bgj_end_time.tv_usec-bgj_start_time.tv_usec)/1000000.0;
        if (sieving_stucked) {
            fprintf(log_err, "get stucked.\n");
        } else if (p->CSD > AMX_MIN_LOG_CSD) {
            fprintf(log_err, "done, time = %.2fs\n", tt);
        }
    }
}

template <uint32_t nb>
void bgj_amx_profile_data_t<nb>::epoch_initial_log(int32_t goal_norm) {
    if (log_level >= 2 && p->CSD > AMX_MIN_LOG_CSD) fprintf(log_out, "epoch %ld, goal_norm = %.2f\n", num_epoch-1, sqrt(2.0 * goal_norm));
}

template <uint32_t nb>
void bgj_amx_profile_data_t<nb>::combine(bgj_amx_profile_data_t<nb> *prof) {
    num_bucket0 += prof->num_bucket0;
    num_bucket1 += prof->num_bucket1;
    num_bucket2 += prof->num_bucket2;
    sum_bucket0_size += prof->sum_bucket0_size;
    sum_bucket1_size += prof->sum_bucket1_size;
    sum_bucket2_size += prof->sum_bucket2_size;
    num_r0 += prof->num_r0;
    num_r1 += prof->num_r1;
    sum_r0_size += prof->sum_r0_size;
    sum_r1_size += prof->sum_r1_size;

    bucket0_time += prof->bucket0_time;
    bucket1_time += prof->bucket1_time;
    bucket2_time += prof->bucket2_time;
    search0_time += prof->search0_time;
    search1_time += prof->search1_time;
    search2_time += prof->search2_time;
    filter_time += prof->filter_time;

    bucket0_ndp += prof->bucket0_ndp;
    bucket1_ndp += prof->bucket1_ndp;
    bucket2_ndp += prof->bucket2_ndp;
    search0_ndp += prof->search0_ndp;
    search1_ndp += prof->search1_ndp;
    search2_ndp += prof->search2_ndp;

    try_add2 += prof->try_add2;
    succ_add2 += prof->succ_add2;
}

template <uint32_t nb>
void bgj_amx_profile_data_t<nb>::one_epoch_log(int bgj) {
    if (log_level >= 1 && p->CSD > AMX_MIN_LOG_CSD) {
        if (bgj == 3) {
            #if BOOST_AMX_SIEVE
            char ftime_str[256];
            char fspeed_str[256];
            sprintf(ftime_str, ", filter: %.3fs", filter_time);
            sprintf(fspeed_str, ", f: %.3f MOPS", succ_add2 / (filter_time + 0.0) / 1048576.0);
            #endif
            double speed0, speed1, speed2;
            fprintf(log_out, "solution collect done, found %ld solutions in %ld buckets, "
                            "b0: %.3fs, b1: %.3fs, b2: %.3fs, s0: %.3fs, s1: %.3fs, s2: %.3fs%s\n",
                     (uint64_t) succ_add2, num_bucket0 + num_bucket1 + num_bucket2 + num_r0 + num_r1, 
                     bucket0_time, bucket1_time, bucket2_time, search0_time, search1_time, search2_time, BOOST_AMX_SIEVE ? ftime_str : "");
            if (log_level >= 2) {
                speed0 = p->CSD * 2 * bucket0_ndp/bucket0_time/1073741824.0/1024.0;
                speed1 = p->CSD * 2 * bucket1_ndp/bucket1_time/1073741824.0/1024.0;
                speed2 = p->CSD * 2 * bucket2_ndp/bucket2_time/1073741824.0/1024.0;
                fprintf(log_out, "b0: %.2f bucket/s (%.3f TFLOPS), b1: %.2f bucket/s (%.3f TFLOPS), b2: %.2f bucket/s (%.3f TFLOPS)\n",
                            num_bucket0/bucket0_time, speed0, num_bucket1/bucket1_time, speed1, num_bucket2/bucket2_time, speed2);
                speed0 = p->CSD * 2 * search0_ndp/search0_time/1073741824.0/1024.0;
                speed1 = p->CSD * 2 * search1_ndp/search1_time/1073741824.0/1024.0;
                speed2 = p->CSD * 2 * search2_ndp/search2_time/1073741824.0/1024.0;
                fprintf(log_out, "s0: %.2f bucket/s (%.3f TFLOPS), s1: %.2f bucket/s (%.3f TFLOPS), "
                                "s2: %.2f bucket/s (%.3f TFLOPS)%s\n",
                                num_bucket0/search0_time, speed0, num_bucket1/search1_time, speed1, 
                                num_bucket2/search2_time, speed2, BOOST_AMX_SIEVE ? fspeed_str : "");
                fprintf(log_out, "nb0 = %ld(%ld), nb1 = %ld(%ld), nb2 = %ld(%ld), nbr0 = %ld(%ld), nbr1 = %ld(%ld)\n", 
                                num_bucket0, (long)(sum_bucket0_size/(0.000001+num_bucket0)), 
                                num_bucket1, (long)(sum_bucket1_size/(0.000001+num_bucket1)), 
                                num_bucket2, (long)(sum_bucket2_size/(0.000001+num_bucket2)),
                                num_r0, (long)(sum_r0_size/(0.000001+num_r0)),
                                num_r1, (long)(sum_r1_size/(0.000001+num_r1)));
                fprintf(log_out, "bucket cost = %ld dp/sol, search cost = %ld dp/sol, try_add2 = %ld, succ_add2 = %ld(1/%.2f), %ld(1/%.2f) inserted\n",
                                (long) ((bucket0_ndp + bucket1_ndp + bucket2_ndp) / (0.000001 + succ_add2)),
                                (long) ((search0_ndp + search1_ndp + search2_ndp) / (0.000001 + succ_add2)),
                                (uint64_t) try_add2, (uint64_t) succ_add2, (try_add2+1e-20)/succ_add2, succ_insert, (succ_add2+1e-20)/succ_insert);
            }
        }
        fflush(log_out);
    }
}

template <uint32_t nb>
void bgj_amx_profile_data_t<nb>::sol_check(sol_list_amx_t **sol_list, long num, int32_t goal_norm, UidHashTable *uid) {
    if (log_level >= 4 && p->CSD > AMX_MIN_LOG_CSD) {
        int pass = 1;
        for (long ind = 0; ind < num; ind++) {
            int passu = 1, passn = 1;
            sol_list_amx_t *sol = sol_list[ind];
            for (long i = 0; i < sol->num_a; i++) {
                uint32_t src1 = sol_list[ind]->a_list[2*i];
                uint32_t src2 = sol_list[ind]->a_list[2*i+1];
                int32_t dp = p->vdpss(src1, src2);
                int32_t ss = p->vnorm[src1] + p->vnorm[src2] + dp;
                uint64_t uu = p->vu[src1] + p->vu[src2];
                if (ss >= goal_norm) {
                    passn = 0;
                }
                #if !BOOST_AMX_SIEVE
                if (!uid->check_uid(uu)) passu = 0;
                #endif
            }
            for (long i = 0; i < sol->num_s; i++) {
                uint32_t src1 = sol_list[ind]->s_list[2*i];
                uint32_t src2 = sol_list[ind]->s_list[2*i+1];
                int32_t dp = p->vdpss(src1, src2);
                int32_t ss = p->vnorm[src1] + p->vnorm[src2] - dp;
                uint64_t uu = p->vu[src1] - p->vu[src2];
                if (ss >= goal_norm) {
                    passn = 0;
                }
                #if !BOOST_AMX_SIEVE
                if (!uid->check_uid(uu)) passu = 0;
                #endif
            }
            if (!passu || !passn) {
                pass = 0;
                fprintf(log_out, "# sol list %ld: %s%s\n", ind, (passn ? "" : "norm failed "), (passu ? "" : "uid failed "));
            }
        }
        #if !BOOST_AMX_SIEVE
        long count = p->num_vec + p->CSD + 1;
        for (long i = 0; i < num; i++) count += sol_list[i]->num_sol();
        long rcount = uid->size();
        if (count != uid->size()) {
            pass = 0;
            fprintf(log_out, "# uid error, %ld vectors, %ld uids in the table\n", count, rcount);
        }
        #endif
        if (pass) fprintf(log_out, "# sol lists verified.\n");
    }
}

template <uint32_t nb>
void bgj_amx_profile_data_t<nb>::pool_bucket_check(bucket_amx_t **bucket_list, long num_bucket, double alpha) {
    if (log_level >= 4 && p->CSD > AMX_MIN_LOG_CSD) {
        int pass = 1;
        __m256 alphax2 = _mm256_set1_ps(alpha * 2.0);
        for (long i = 0; i < num_bucket; i++) {
            int passn = 1, passdp = 1;
            bucket_amx_t *bkt = bucket_list[i];
            for (long j = 0; j < bkt->num_pvec; j++) {
                uint32_t ind = bkt->pvec[j];
                uint32_t norm = bkt->pnorm[j];
                int32_t bound = _mm256_extract_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(alphax2, _mm256_cvtepi32_ps(_mm256_set1_epi32(norm)))), 0);
                int32_t rdp = 0;
                for (long l = 0; l < p->CSD; l++) 
                    rdp += (int) (p->vec + bkt->center_ind * p->vec_length)[l] * (int) (p->vec + ind * p->vec_length)[l];
                if (abs(rdp) <= bound) passdp = 0;
                if (p->vnorm[ind] != norm) passn = 0;
            }
            for (long j = 0; j < bkt->num_nvec; j++) {
                uint32_t ind = bkt->nvec[j];
                uint32_t norm = bkt->nnorm[j];
                int32_t rdp = 0;
                int32_t bound = _mm256_extract_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(alphax2, _mm256_cvtepi32_ps(_mm256_set1_epi32(norm)))), 0);
                for (long l = 0; l < p->CSD; l++) 
                    rdp += (int) (p->vec + bkt->center_ind * p->vec_length)[l] * (int) (p->vec + ind * p->vec_length)[l];
                if (abs(rdp) <= bound) passdp = 0;
                if (p->vnorm[ind] != norm) passn = 0;
            }
            if (!passn || !passdp) {
                pass = 0;
                fprintf(log_out, "# bucket %ld: %s%s\n", i, (passn ? "" : "norm failed "), (passdp ? "" : "dp failed "));
            }
        }
        long *count = (long *) NEW_VEC(num_bucket, sizeof(long));
        for (long i = 0; i < p->num_vec; i++) {
            for (long j = 0; j < num_bucket; j++) {
                int32_t rdp = 0;
                int32_t norm = p->vnorm[i];
                int32_t bound = _mm256_extract_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(alphax2, _mm256_cvtepi32_ps(_mm256_set1_epi32(norm)))), 0);
                for (long l = 0; l < p->vec_length; l++) 
                    rdp += (int) (p->vec + bucket_list[j]->center_ind * p->vec_length)[l] * 
                            (int) (p->vec + i * p->vec_length)[l];
                if (abs(rdp) > bound) count[j]++;
            }
        }
        for (long i = 0; i < num_bucket; i++) {
            if (count[i] - 1 > bucket_list[i]->num_pvec + bucket_list[i]->num_nvec) {
                pass = 0;
                fprintf(log_out, "# bucket %ld, found %ld / %ld\n", i, bucket_list[i]->num_pvec + bucket_list[i]->num_nvec, count[i]);
            }
        }
        FREE_VEC((void *)count);
        
        if (pass) fprintf(log_out, "# buckets verified.\n");
    }
}

template <uint32_t nb>
void bgj_amx_profile_data_t<nb>::subbucket_check(bucket_amx_t **bucket_list, bucket_amx_t **subbucket_list, long num_bucket, long num_subbucket, double alpha) {
    if (log_level >= 4 && p->CSD > AMX_MIN_LOG_CSD) {
        int pass = 1;
        long num_null = 0;
        __m256 alphax2 = _mm256_set1_ps(alpha * 2.0);
        for (long k = 0; k < num_bucket; k++) {
            for (long i = k * num_subbucket; i < (k + 1) * num_subbucket; i++) {
                bucket_amx_t *bkt = bucket_list[k];
                bucket_amx_t *sbkt = subbucket_list[i];
                if (sbkt == NULL || sbkt->num_nvec + sbkt->num_pvec == 0) {
                    num_null++;
                    continue;
                }
                int passn = 1, passdp = 1;
                uint32_t cind = sbkt->center_ind;
                for (long j = 0; j < sbkt->num_pvec; j++) {
                    uint32_t ind = sbkt->pvec[j];
                    uint32_t norm = sbkt->pnorm[j];
                    int32_t bound = _mm256_extract_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(alphax2, _mm256_cvtepi32_ps(_mm256_set1_epi32(norm)))), 0);
                    int32_t rdp = 0;
                    for (long l = 0; l < p->CSD; l++) 
                        rdp += (int) (p->vec + cind * p->vec_length)[l] * (int) (p->vec + ind * p->vec_length)[l];
                    if (abs(rdp) <= bound) passdp = 0;
                    if (p->vnorm[ind] != norm) passn = 0;
                }
                for (long j = 0; j < sbkt->num_nvec; j++) {
                    uint32_t ind = sbkt->nvec[j];
                    uint32_t norm = sbkt->nnorm[j];
                    int32_t bound = _mm256_extract_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(alphax2, _mm256_cvtepi32_ps(_mm256_set1_epi32(norm)))), 0);
                    int32_t rdp = 0;
                    for (long l = 0; l < p->CSD; l++) 
                        rdp += (int) (p->vec + cind * p->vec_length)[l] * (int) (p->vec + ind * p->vec_length)[l];
                    if (abs(rdp) <= bound) passdp = 0;
                    if (p->vnorm[ind] != norm) passn = 0;
                }
                
                if (!passn || !passdp) {
                    pass = 0;
                    fprintf(log_out, "# bucket %ld: %s%s\n", i, (passn ? "" : "norm failed "), (passdp ? "" : "dp failed "));
                }

                long countp = 0, countn = 0;
                for (long i = 0; i < bkt->num_pvec; i++) {
                    int32_t rdp = 0;
                    int32_t norm = bkt->pnorm[i];
                    int32_t bound = _mm256_extract_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(alphax2, _mm256_cvtepi32_ps(_mm256_set1_epi32(norm)))), 0);
                    for (long l = 0; l < p->vec_length; l++) 
                        rdp += (int) (p->vec + sbkt->center_ind * p->vec_length)[l] * 
                                (int) (p->vec + bkt->pvec[i] * p->vec_length)[l];
                    if (rdp > bound) countp++;
                }
                for (long i = 0; i < bkt->num_nvec; i++) {
                    int32_t rdp = 0;
                    int32_t norm = bkt->nnorm[i];
                    int32_t bound = _mm256_extract_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(alphax2, _mm256_cvtepi32_ps(_mm256_set1_epi32(norm)))), 0);
                    for (long l = 0; l < p->vec_length; l++) 
                        rdp += (int) (p->vec + sbkt->center_ind * p->vec_length)[l] * 
                                (int) (p->vec + bkt->nvec[i] * p->vec_length)[l];
                    if (-rdp > bound) countn++;
                }
                if (countp - 1 > sbkt->num_pvec) {
                    pass = 0;
                    fprintf(log_out, "# subbucket %ld of bucket %ld, found %ld / %ld pvec\n", i, k, sbkt->num_pvec, countp-1);
                }
                if (countn > sbkt->num_nvec) {
                    pass = 0;
                    fprintf(log_out, "# subbucket %ld of bucket %ld, found %ld / %ld nvec\n", i, k, sbkt->num_nvec, countn);
                }
            }
        }
        if (num_null) fprintf(log_out, "# warning: %ld of %ld buckets duplicated\n", num_null, num_subbucket * num_bucket);
        if (pass) fprintf(log_out, "# buckets verified.\n");
    }
}

template <uint32_t nb>
void bgj_amx_profile_data_t<nb>::insert_log(uint64_t num_total_sol, double insert_time) {
    if (log_level >= 2 && p->CSD > AMX_MIN_LOG_CSD){
        fprintf(log_out, "insert %ld solutions in %fs\n", num_total_sol, insert_time);
        fflush(log_out);
    }
    if (log_level == 0 && p->CSD > AMX_MIN_LOG_CSD){
        fprintf(log_err, ".");
    }
}

template <uint32_t nb>
void bgj_amx_profile_data_t<nb>::insert_inner_log(uint64_t *length_stat, uint64_t num_linfty_failed, uint64_t num_l2_failed, uint64_t num_not_try) {
    if (p->CSD <= AMX_MIN_LOG_CSD) return;
    fprintf(log_out, "length_stat = [");
    for (long i = 90; i < 110; i++) fprintf(log_out, "%lu ", length_stat[i]);
    fprintf(log_out, "%lu]\n", length_stat[110]);
    fprintf(log_out, "num_linfty_failed = %lu, num_l2_failed = %lu, num_not_try = %lu\n", num_linfty_failed, num_l2_failed, num_not_try);
    fflush(log_out);
}

template <uint32_t nb>
void bgj_amx_profile_data_t<nb>::report_bucket_not_used(int bgj, long nrem0, long nrem1, long nrem2) {
    if (log_level >= 1 && p->CSD > AMX_MIN_LOG_CSD) {
        if (bgj == 3) {
            if (nrem0) fprintf(log_out, "bucket0: %ld buckets not used\n", nrem0);
            if (nrem1) fprintf(log_out, "bucket1: %ld buckets not used\n", nrem1);
            if (nrem2) fprintf(log_out, "bucket2: %ld buckets not used\n", nrem2);
            fflush(log_out);
        }
    }
}

template <uint32_t nb>
int Pool_epi8_t<nb>::bgj3_Sieve_amx(long log_level, long max_epoch, double goal_norm_scale, long ESD, double prefer_deep) {
    if (nb != 5) {
        printf("[Error] Pool_epi8_t<nb>::bgj3_Sieve_amx: nb != 5 unsupported\n");
        return -1;
    }

    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        printf("[Error] Pool_epi8_t<%u>::bgj3_Sieve_amx: Fail to do XFEATURE_XTILEDATA, nothing done.\n", nb);
        return -1;
    }

    #if BOOST_AMX_SIEVE
    if (booster == NULL) {
        booster = new booster_amx160_t;
    }
    booster->init(this, ESD, prefer_deep);
    booster->reconstruct_all_score();
    sorted_index = 0;
    #endif

    bgj_amx_profile_data_t<nb> main_profile;
    main_profile.init(this, log_level);

    ///////////////// params /////////////////
    double alpha_r0 = BGJ3_AMX_REUSE0_ALPHA;
    double alpha_r1 = BGJ3_AMX_REUSE1_ALPHA;
    double alpha_b0 = BGJ3_AMX_BUCKET0_ALPHA;
    double alpha_b1 = BGJ3_AMX_BUCKET1_ALPHA;
    double alpha_b2 = BGJ3_AMX_BUCKET2_ALPHA;
    const double saturation_radius = 4.0/3.0;
    #if BOOST_AMX_SIEVE
    double saturation_ratio = AMX_BOOST_SATURATION_RATIO;
    #else
    double saturation_ratio = 0.375;
    #endif
    const double one_epoch_ratio = 0.025;
    const double improve_ratio = 0.73;
    const double resort_ratio = 0.95;

    do {
        std::ifstream VINstream(".vin");
        double sr = 0.0;
        VINstream >> sr;
        if (sr > 0.01 && sr < 1.0) saturation_ratio = sr;
        VINstream.close();
    } while (0);

    ///////////////// sort before sieve /////////////////
    TIMER_START;
    sort_cvec();
    TIMER_END;
    main_profile.sort_time += CURRENT_TIME;

    main_profile.initial_log(3);

    long sieving_stucked = 0;
    double first_collect_sol = -1.0;

    ///////////////// main sieving procedure /////////////////
    while (!sieve_is_over(saturation_radius, saturation_ratio) && !sieving_stucked) {
        main_profile.num_epoch++;
        const long goal_index = (long)(improve_ratio * num_vec);
        int32_t _goal_norm;
        for (long gind = goal_index; gind >= 0; gind--) {
            int32_t rn = vnorm[*((uint32_t *)(cvec + 3LL * gind))];
            int32_t cn = cvec[3LL * gind + 2LL];
            int32_t expcn = rn >> 1;
            expcn = (expcn > 65535) ? 65535 : expcn;
            if (abs(expcn - cn) <= 2) {
                _goal_norm = rn;
                break;
            }
            if (gind + (num_vec >> 5) < goal_index) {
                fprintf(stderr, "[Warning] Pool_epi8_t<%u>::bgj3_Sieve_amx: goal_norm not found, use score instead.\n", nb);
                _goal_norm = cvec[3LL * goal_index + 2LL];
                _goal_norm *= 2;
                break;
            }
        } 
        const int32_t goal_norm = _goal_norm * (goal_norm_scale == -1.0 ? AMX_BOOST_GOAL_NORM_SCALE : goal_norm_scale);
        const int32_t goal_score = cvec[3LL * goal_index + 2];
        main_profile.epoch_initial_log(goal_norm);

        bgj_amx_profile_data_t<nb> local_profile;
        local_profile.init(this, log_level);

        ///////////////// collect solutions /////////////////
        long stucktime = 0;
        long num_total_sol = 0;
        long last_num_total_sol = 0;
        sol_list_amx_t *sol_list[AMX_MAX_NTHREADS];
        for (long i = 0; i < num_threads; i++) sol_list[i] = new sol_list_amx_t;
        bool rel_collection_stop = false;
        do {
            ///////////// bucket0 //////////////
            bucket_amx_t *rbucket0[BGJ3_AMX_BUCKET0_BATCHSIZE] = {};
            bucket_amx_t *bucket0[BGJ3_AMX_BUCKET0_BATCHSIZE] = {};

            TIMER_START;           
            _pool_bucketing_amx<BGJ3_AMX_BUCKET0_BATCHSIZE, BGJ3_AMX_USE_FARAWAY_CENTER, 1>(rbucket0, bucket0, alpha_r0, alpha_b0, sol_list, goal_norm, &local_profile);
            TIMER_END;
            local_profile.bucket0_time += CURRENT_TIME;
            local_profile.num_bucket0 += BGJ3_AMX_BUCKET0_BATCHSIZE;
            local_profile.num_r0 += BGJ3_AMX_BUCKET0_BATCHSIZE;
            local_profile.bucket0_ndp += num_vec * BGJ3_AMX_BUCKET0_BATCHSIZE;
            for (long i = 0; i < BGJ3_AMX_BUCKET0_BATCHSIZE; i++) {
                local_profile.sum_bucket0_size += bucket0[i]->num_nvec+bucket0[i]->num_pvec;
                local_profile.sum_r0_size += rbucket0[i]->num_nvec+rbucket0[i]->num_pvec;
            }
            local_profile.pool_bucket_check(rbucket0, BGJ3_AMX_BUCKET0_BATCHSIZE, alpha_r0);
            local_profile.pool_bucket_check(bucket0, BGJ3_AMX_BUCKET0_BATCHSIZE, alpha_b0);
            local_profile.sol_check(sol_list, num_threads, goal_norm, uid);

            #if AMX_PARALLEL_BUCKET1
            ///////////// bucket1 //////////////
            // these buckets will be deleted after the for loop
            bucket_amx_t *rbucket1[BGJ3_AMX_BUCKET1_BATCHSIZE] = {};
            bucket_amx_t *bucket1[BGJ3_AMX_BUCKET1_BATCHSIZE] = {};
            bucket_amx_t *bucket2[BGJ3_AMX_BUCKET1_BATCHSIZE * BGJ3_AMX_BUCKET2_BATCHSIZE] = {};
            // rbucket0 and bucket0 should be deleted after used
            int rbucket0_nrem = BGJ3_AMX_BUCKET0_BATCHSIZE;
            int rbucket1_nrem;
            int bucket2_nrem[BGJ3_AMX_BUCKET1_BATCHSIZE];
            pthread_spinlock_t bucket_list_lock;
            pthread_spin_init(&bucket_list_lock, PTHREAD_PROCESS_SHARED);
            int too_many_sol = 0;
            
            for (long ind_b0 = 0; ind_b0 < BGJ3_AMX_BUCKET0_BATCHSIZE; ind_b0++) {
                if (too_many_sol) break;
                for (long i = 0; i < BGJ3_AMX_BUCKET1_BATCHSIZE; i++) bucket2_nrem[i] = -1;
                rbucket1_nrem = 0;
                TIMER_START;
                _parallel_sub_bucketing_amx<BGJ3_AMX_BUCKET1_BATCHSIZE, BGJ3_AMX_USE_FARAWAY_CENTER, 1>(
                    bucket0[ind_b0], rbucket1, bucket1, alpha_r1, alpha_b1, sol_list, goal_norm, &local_profile);
                TIMER_END;
                local_profile.bucket1_time += CURRENT_TIME;
                local_profile.bucket1_ndp += (bucket0[ind_b0]->num_pvec + bucket0[ind_b0]->num_nvec) * BGJ3_AMX_BUCKET1_BATCHSIZE;
                for (long i = 0; i < BGJ3_AMX_BUCKET1_BATCHSIZE; i++) {
                    if (rbucket1[i]->num_pvec + rbucket1[i]->num_nvec) {
                        local_profile.num_r1++;
                        local_profile.num_bucket1++;
                        local_profile.sum_r1_size += rbucket1[i]->num_nvec+rbucket1[i]->num_pvec;
                        local_profile.sum_bucket1_size += bucket1[i]->num_nvec+bucket1[i]->num_pvec;
                    }
                }
                for (long i = 0; i < BGJ3_AMX_BUCKET1_BATCHSIZE; i++) {
                    rbucket1_nrem += (rbucket1[i]->num_pvec + rbucket1[i]->num_nvec) ? 1 : 0;
                }

                local_profile.subbucket_check(bucket0+ind_b0, bucket1, 1, BGJ3_AMX_BUCKET1_BATCHSIZE, alpha_b1); 
                local_profile.subbucket_check(bucket0+ind_b0, rbucket1, 1, BGJ3_AMX_BUCKET1_BATCHSIZE, alpha_r1);
                local_profile.sol_check(sol_list, num_threads, goal_norm, uid);
                
                #pragma omp parallel for
                for (long thread = 0; thread < num_threads; thread++) {
                    const long begin_ind = (thread * BGJ3_AMX_BUCKET1_BATCHSIZE) / num_threads;
                    const long end_ind = ((thread + 1) * BGJ3_AMX_BUCKET1_BATCHSIZE) / num_threads;
                    for (long ind = begin_ind; ind < end_ind; ind++) {
                        if (bucket1[ind]->num_pvec + bucket1[ind]->num_nvec == 0) continue;

                        TIMER_START;
                        _sub_bucketing_amx<BGJ3_AMX_BUCKET2_BATCHSIZE, BGJ3_AMX_USE_FARAWAY_CENTER, 0>(
                            bucket1[ind], NULL, bucket2 + ind * BGJ3_AMX_BUCKET2_BATCHSIZE, 0.0, alpha_b2, sol_list[thread], goal_norm, &local_profile);
                        TIMER_END;
                        pthread_spin_lock(&local_profile.profile_lock);
                        local_profile.bucket2_time += CURRENT_TIME;
                        local_profile.bucket2_ndp += (bucket1[ind]->num_pvec + bucket1[ind]->num_nvec) * BGJ3_AMX_BUCKET2_BATCHSIZE;
                        for (long i = 0; i < BGJ3_AMX_BUCKET2_BATCHSIZE; i++) {
                            if (bucket2[ind * BGJ3_AMX_BUCKET2_BATCHSIZE + i]->num_pvec + bucket2[ind * BGJ3_AMX_BUCKET2_BATCHSIZE + i]->num_nvec) {
                                local_profile.num_bucket2++;
                                local_profile.sum_bucket2_size += bucket2[ind * BGJ3_AMX_BUCKET2_BATCHSIZE + i]->num_nvec+bucket2[ind * BGJ3_AMX_BUCKET2_BATCHSIZE + i]->num_pvec;
                            }
                        }
                        pthread_spin_unlock(&local_profile.profile_lock);
                        
                        local_profile.subbucket_check(bucket1+ind, bucket2 + ind * BGJ3_AMX_BUCKET2_BATCHSIZE, 1, BGJ3_AMX_BUCKET2_BATCHSIZE, alpha_b2);
                        bucket1[ind]->num_pvec = 0;
                        bucket1[ind]->num_nvec = 0;
                        
                        int nb2 = 0;
                        for (long i = 0; i < BGJ3_AMX_BUCKET2_BATCHSIZE; i++) {
                            if (bucket2[ind * BGJ3_AMX_BUCKET2_BATCHSIZE + i]->num_pvec + bucket2[ind * BGJ3_AMX_BUCKET2_BATCHSIZE + i]->num_nvec) nb2++;
                        }
                        bucket2_nrem[ind] = nb2;
                    }

                    for (;;) {
                        bucket_amx_t *bkt = NULL;
                        int type = 0;
                        int idx = -1;
                        int finished = 1;
                        pthread_spin_lock(&bucket_list_lock);
                        if (rbucket0_nrem) {
                            bkt = rbucket0[rbucket0_nrem - 1];
                            rbucket0[rbucket0_nrem - 1] = NULL;
                            rbucket0_nrem--;
                        } else if (rbucket1_nrem) {
                            for (long i = 0; i < BGJ3_AMX_BUCKET1_BATCHSIZE; i++) {
                                if (rbucket1[i] == NULL) continue;
                                if (rbucket1[i]->num_pvec + rbucket1[i]->num_nvec) {
                                    bkt = rbucket1[i];
                                    rbucket1[i] = NULL;
                                    rbucket1_nrem--;
                                    idx = i;
                                    type = 1;
                                    break;
                                }
                            }
                        } else {
                            for (long i = 0; i < BGJ3_AMX_BUCKET1_BATCHSIZE; i++) {
                                if (bucket2_nrem[i] > 0) {
                                    for (long j = 0; j < BGJ3_AMX_BUCKET2_BATCHSIZE; j++) {
                                        if (bucket2[i * BGJ3_AMX_BUCKET2_BATCHSIZE + j] == NULL) continue;
                                        if (bucket2[i * BGJ3_AMX_BUCKET2_BATCHSIZE + j]->num_pvec + bucket2[i * BGJ3_AMX_BUCKET2_BATCHSIZE + j]->num_nvec) {
                                            bkt = bucket2[i * BGJ3_AMX_BUCKET2_BATCHSIZE + j];
                                            bucket2[i * BGJ3_AMX_BUCKET2_BATCHSIZE + j] = NULL;
                                            bucket2_nrem[i]--;
                                            idx = i * BGJ3_AMX_BUCKET2_BATCHSIZE + j;
                                            type = 2;
                                            break;
                                        }
                                    }
                                    break;
                                } else if (bucket2_nrem[i] == -1) {
                                    finished = 0;
                                }
                            }
                            if (finished && bkt == NULL) {
                                pthread_spin_unlock(&bucket_list_lock);
                                break;
                            }
                        }
                        pthread_spin_unlock(&bucket_list_lock);
                        
                        if (bkt == NULL) continue;

                        TIMER_START;
                        _search_amx(bkt, sol_list[thread], goal_norm, &local_profile);
                        TIMER_END;
                        if (sol_list[thread]->num_a + sol_list[thread]->num_s > sol_list[thread]->num_a_insert + sol_list[thread]->num_s_insert + 1048576) {
                            booster->filter_sol_list(sol_list[thread], goal_score);
                        }
                        pthread_spin_lock(&local_profile.profile_lock);
                        if (type == 2) {
                            local_profile.search2_time += CURRENT_TIME;
                            local_profile.search2_ndp += (bkt->num_pvec + bkt->num_nvec) * (bkt->num_pvec + bkt->num_nvec - 1) / 2;
                        } else if (type == 1) {
                            local_profile.search1_time += CURRENT_TIME;
                            local_profile.search1_ndp += (bkt->num_pvec + bkt->num_nvec) * (bkt->num_pvec + bkt->num_nvec - 1) / 2;
                        } else {
                            local_profile.search0_time += CURRENT_TIME;
                            local_profile.search0_ndp += (bkt->num_pvec + bkt->num_nvec) * (bkt->num_pvec + bkt->num_nvec - 1) / 2;
                        }
                        pthread_spin_unlock(&local_profile.profile_lock);
                        bkt->num_pvec = 0;
                        bkt->num_nvec = 0;
                        if (type == 0) delete bkt;
                        else {
                            bkt->num_pvec = 0;
                            bkt->num_nvec = 0;
                            pthread_spin_lock(&bucket_list_lock);
                            if (type == 1) {
                                rbucket1[idx] = bkt;
                            } else {
                                bucket2[idx] = bkt;
                            }
                            pthread_spin_unlock(&bucket_list_lock);
                        }
                        long already_found = 0;
                        #if BOOST_AMX_SIEVE
                        for (long i = 0; i < num_threads; i++) {
                            already_found += sol_list[i]->num_a_insert + sol_list[i]->num_s_insert;
                        }
                        #else
                        already_found = local_profile.succ_add2;
                        #endif
                        if (already_found > num_empty + sorted_index - goal_index) {
                            pthread_spin_lock(&local_profile.profile_lock);
                            if (too_many_sol) {
                                pthread_spin_unlock(&local_profile.profile_lock);
                                break;
                            }
                            too_many_sol = 1;
                            long b0_not_used = BGJ3_AMX_BUCKET0_BATCHSIZE - 1 - ind_b0;
                            long b1_not_used = 0;
                            long b2_not_used = rbucket1_nrem + rbucket0_nrem;
                            for (long i = 0; i < BGJ3_AMX_BUCKET1_BATCHSIZE; i++) {
                                if (bucket2_nrem[i] == -1 || bucket2_nrem[i] == BGJ3_AMX_BUCKET2_BATCHSIZE) {
                                    b1_not_used++;
                                } else if (bucket2_nrem[i] > 0) {
                                    b2_not_used += bucket2_nrem[i];
                                }
                            }
                            local_profile.report_bucket_not_used(3, b0_not_used, b1_not_used, b2_not_used);
                            pthread_spin_unlock(&local_profile.profile_lock);
                            break;
                        }
                        
                    }
                }

                delete bucket0[ind_b0];
                bucket0[ind_b0] = NULL;
            }
            
            for (long i = 0; i < BGJ3_AMX_BUCKET0_BATCHSIZE; i++) {
                if (bucket0[i]) delete bucket0[i];
                if (rbucket0[i]) delete rbucket0[i];
            }
            #pragma omp parallel for
            for (long i = 0; i < BGJ3_AMX_BUCKET1_BATCHSIZE; i++) {
                if (bucket1[i]) delete bucket1[i];
                if (rbucket1[i]) delete rbucket1[i];
                for (long j = 0; j < BGJ3_AMX_BUCKET2_BATCHSIZE; j++) {
                    if (bucket2[i * BGJ3_AMX_BUCKET2_BATCHSIZE + j]) delete bucket2[i * BGJ3_AMX_BUCKET2_BATCHSIZE + j];
                }
            }
            #else 
            // copied from bgj3_Sieve_epi8
            bucket_amx_t *rbucket1[BGJ3_AMX_BUCKET0_BATCHSIZE * BGJ3_AMX_BUCKET1_BATCHSIZE] = {};
            bucket_amx_t *bucket1[BGJ3_AMX_BUCKET0_BATCHSIZE * BGJ3_AMX_BUCKET1_BATCHSIZE] = {};
            bucket_amx_t *bucket2[AMX_MAX_NTHREADS * BGJ3_AMX_BUCKET2_BATCHSIZE] = {};
            int rbucket1_nrem[BGJ3_AMX_BUCKET0_BATCHSIZE];
            int rbucket0_nrem = BGJ3_AMX_BUCKET0_BATCHSIZE;
            int too_many_sol = 0;
            for (long i = 0; i < BGJ3_AMX_BUCKET0_BATCHSIZE; i++) rbucket1_nrem[i] = -1;
            pthread_spinlock_t bucket_list_lock;
            pthread_spin_init(&bucket_list_lock, PTHREAD_PROCESS_SHARED);
            #pragma omp parallel for
            for (long thread = 0; thread < num_threads; thread++) {
                // bucket1
                do {
                    const long begin_ind = (thread * BGJ3_AMX_BUCKET0_BATCHSIZE) / num_threads;
                    const long end_ind = ((thread + 1) * BGJ3_AMX_BUCKET0_BATCHSIZE) / num_threads;
                    long ndp = 0;
                    TIMER_START;
                    for (long ind = begin_ind; ind < end_ind; ind++) {
                        _sub_bucketing_amx<BGJ3_AMX_BUCKET1_BATCHSIZE, BGJ3_AMX_USE_FARAWAY_CENTER, 1>(
                            bucket0[ind], &rbucket1[ind * BGJ3_AMX_BUCKET1_BATCHSIZE], &bucket1[ind * BGJ3_AMX_BUCKET1_BATCHSIZE], 
                            alpha_r1, alpha_b1, sol_list[thread], goal_norm, &local_profile);
                        ndp += BGJ3_AMX_BUCKET1_BATCHSIZE * (bucket0[ind]->num_pvec + bucket0[ind]->num_nvec);
                        if (local_profile.log_level <= 3) delete bucket0[ind];
                        long num_bucket1_done = 0;
                        for (long j = 0; j < BGJ3_AMX_BUCKET1_BATCHSIZE; j++) {
                            if (bucket1[ind * BGJ3_AMX_BUCKET1_BATCHSIZE + j]->num_pvec + bucket1[ind * BGJ3_AMX_BUCKET1_BATCHSIZE + j]->num_nvec) num_bucket1_done++;
                            else {
                                delete bucket1[ind * BGJ3_AMX_BUCKET1_BATCHSIZE + j];
                                bucket1[ind * BGJ3_AMX_BUCKET1_BATCHSIZE + j] = NULL;
                                delete rbucket1[ind * BGJ3_AMX_BUCKET1_BATCHSIZE + j];
                                rbucket1[ind * BGJ3_AMX_BUCKET1_BATCHSIZE + j] = NULL;
                            }
                        }
                        rbucket1_nrem[ind] = num_bucket1_done;
                    }
                    TIMER_END;
                    pthread_spin_lock(&local_profile.profile_lock);
                    local_profile.bucket1_time += CURRENT_TIME;
                    local_profile.bucket1_ndp += ndp;
                    for (long i = begin_ind; i < end_ind; i++) {
                        for (long j = 0; j < BGJ3_AMX_BUCKET1_BATCHSIZE; j++) {
                            if (rbucket1[i * BGJ3_AMX_BUCKET1_BATCHSIZE + j]) {
                                local_profile.num_r1++;
                                local_profile.sum_r1_size += rbucket1[i * BGJ3_AMX_BUCKET1_BATCHSIZE + j]->num_pvec + rbucket1[i * BGJ3_AMX_BUCKET1_BATCHSIZE + j]->num_nvec;
                            }
                            if (bucket1[i * BGJ3_AMX_BUCKET1_BATCHSIZE + j]) {
                                local_profile.num_bucket1++;
                                local_profile.sum_bucket1_size += bucket1[i * BGJ3_AMX_BUCKET1_BATCHSIZE + j]->num_pvec + bucket1[i * BGJ3_AMX_BUCKET1_BATCHSIZE + j]->num_nvec;
                            }
                        }
                    }
                    local_profile.subbucket_check(bucket0+begin_ind, bucket1 + begin_ind * BGJ3_AMX_BUCKET1_BATCHSIZE, 
                                end_ind - begin_ind, BGJ3_AMX_BUCKET1_BATCHSIZE, alpha_b1);
                    local_profile.subbucket_check(bucket0+begin_ind, rbucket1 + begin_ind * BGJ3_AMX_BUCKET1_BATCHSIZE, 
                                end_ind - begin_ind, BGJ3_AMX_BUCKET1_BATCHSIZE, alpha_r1);
                    local_profile.sol_check(sol_list, num_threads, goal_norm, uid);
                    pthread_spin_unlock(&local_profile.profile_lock);
                    if (local_profile.log_level >= 4) {
                        for (long ind = begin_ind; ind < end_ind; ind++) delete bucket0[ind];
                    }
                } while (0);
                
                for (;;) {
                    int type = -1;
                    bucket_amx_t *bkt3 = NULL;
                    bucket_amx_t *bkt2 = NULL;
                    bucket_amx_t *bkt = NULL;
                    pthread_spin_lock(&bucket_list_lock);
                    if (rbucket0_nrem > 0) {
                        bkt3 = rbucket0[rbucket0_nrem-1];
                        rbucket0[rbucket0_nrem-1] = NULL;
                        rbucket0_nrem--;
                        type = 0;
                    } else {
                        for (long i = 0; i < BGJ3_AMX_BUCKET0_BATCHSIZE; i++) {
                            if (rbucket1_nrem[i] > 0) {
                                for (long j = 0; j < BGJ3_AMX_BUCKET1_BATCHSIZE; j++) {
                                    if (bucket1[i * BGJ3_AMX_BUCKET1_BATCHSIZE + j]) {
                                        bkt2 = rbucket1[i * BGJ3_AMX_BUCKET1_BATCHSIZE + j];
                                        bkt = bucket1[i * BGJ3_AMX_BUCKET1_BATCHSIZE + j];
                                        bucket1[i * BGJ3_AMX_BUCKET1_BATCHSIZE + j] = NULL;
                                        rbucket1[i * BGJ3_AMX_BUCKET1_BATCHSIZE + j] = NULL;
                                        rbucket1_nrem[i]--;
                                        type = 1;
                                        break;
                                    }
                                }
                                break;
                            }
                        }
                        if (type == -1) {
                            int finished = 1;
                            for (long i = 0; i < BGJ3_AMX_BUCKET0_BATCHSIZE; i++) {
                                if (rbucket1_nrem[i] == -1) {
                                    finished = 0;
                                    break;
                                }
                            }
                            if (finished) {
                                pthread_spin_unlock(&bucket_list_lock);
                                break;
                            }
                        }
                    }
                    pthread_spin_unlock(&bucket_list_lock);
                    if (type == -1) continue;
                    if (type == 0) {
                        TIMER_START;
                        _search_amx(bkt3, sol_list[thread], goal_norm, &local_profile);
                        TIMER_END;
                        if (sol_list[thread]->num_a + sol_list[thread]->num_s > sol_list[thread]->num_a_insert + sol_list[thread]->num_s_insert + 1048576) {
                            booster->filter_sol_list(sol_list[thread], goal_score);
                        }
                        pthread_spin_lock(&local_profile.profile_lock);
                        local_profile.search0_time += CURRENT_TIME;
                        local_profile.search0_ndp += (bkt3->num_pvec + bkt3->num_nvec) * (bkt3->num_pvec + bkt3->num_nvec - 1) / 2;
                        pthread_spin_unlock(&local_profile.profile_lock);
                        delete bkt3;
                    } else {
                        long __search1_ndp = 0;
                        double __search1_time = 0.0;

                        double __bucket2_time = 0.0;
                        long __bucket2_ndp = 0;
                        long __sum_bucket2_size = 0;
                        long __num_bucket2 = 0;
                        long __search2_ndp = 0;
                        double __search2_time = 0.0;
                        
                        TIMER_START;
                        _search_amx(bkt2, sol_list[thread], goal_norm, &local_profile);                 
                        TIMER_END;
                        if (sol_list[thread]->num_a + sol_list[thread]->num_s > sol_list[thread]->num_a_insert + sol_list[thread]->num_s_insert + 1048576) {
                            booster->filter_sol_list(sol_list[thread], goal_score);
                        }
                        __search1_ndp += (bkt2->num_pvec + bkt2->num_nvec) * (bkt2->num_pvec + bkt2->num_nvec - 1) / 2;
                        __search1_time += CURRENT_TIME;
                        delete bkt2;

                        if (too_many_sol) {
                            pthread_spin_lock(&local_profile.profile_lock);
                            local_profile.search1_time += __search1_time;
                            local_profile.search1_ndp += __search1_ndp;
                            pthread_spin_unlock(&local_profile.profile_lock);
                            delete bkt;
                            break;
                        }

                        TIMER_START;
                        _sub_bucketing_amx<BGJ3_AMX_BUCKET2_BATCHSIZE, BGJ3_AMX_USE_FARAWAY_CENTER, 0>(
                            bkt, NULL, bucket2 + thread * BGJ3_AMX_BUCKET2_BATCHSIZE, 0.0, alpha_b2, sol_list[thread], goal_norm, &local_profile);
                        TIMER_END;
                        __bucket2_time += CURRENT_TIME;
                        __bucket2_ndp = BGJ3_AMX_BUCKET2_BATCHSIZE * (bkt->num_pvec + bkt->num_nvec);
                        for (long i = 0; i < BGJ3_AMX_BUCKET2_BATCHSIZE; i++) {
                            if (bucket2[thread * BGJ3_AMX_BUCKET2_BATCHSIZE + i]->num_pvec + bucket2[thread * BGJ3_AMX_BUCKET2_BATCHSIZE + i]->num_nvec) {
                                __sum_bucket2_size += bucket2[thread * BGJ3_AMX_BUCKET2_BATCHSIZE + i]->num_pvec + bucket2[thread * BGJ3_AMX_BUCKET2_BATCHSIZE + i]->num_nvec;
                                __num_bucket2++;
                            }
                        }
                        TIMER_START;
                        for (long i = 0; i < BGJ3_AMX_BUCKET2_BATCHSIZE; i++) {
                            if (too_many_sol) break;
                            bucket_amx_t *_bkt = bucket2[thread * BGJ3_AMX_BUCKET2_BATCHSIZE + i];
                            if (_bkt->num_pvec + _bkt->num_nvec == 0) continue;
                            _search_amx(_bkt, sol_list[thread], goal_norm, &local_profile);
                            double ft = 0.0;
                            if (sol_list[thread]->num_a + sol_list[thread]->num_s > sol_list[thread]->num_a_insert + sol_list[thread]->num_s_insert + 1048576) {
                                ft = booster->filter_sol_list(sol_list[thread], goal_score);
                            }
                            __search2_ndp += (_bkt->num_pvec + _bkt->num_nvec) * (_bkt->num_pvec + _bkt->num_nvec - 1) / 2;
                            _bkt->num_pvec = 0;
                            _bkt->num_nvec = 0;
                            __search2_time -= ft;
                        }
                        TIMER_END;
                        __search2_time += CURRENT_TIME;

                        pthread_spin_lock(&local_profile.profile_lock);
                        local_profile.search1_time += __search1_time;
                        local_profile.search2_time += __search2_time;
                        local_profile.bucket2_time += __bucket2_time;
                        local_profile.search1_ndp += __search1_ndp;
                        local_profile.search2_ndp += __search2_ndp;
                        local_profile.bucket2_ndp += __bucket2_ndp;
                        local_profile.sum_bucket2_size += __sum_bucket2_size;
                        local_profile.num_bucket2 += __num_bucket2;
                        pthread_spin_unlock(&local_profile.profile_lock);
                        delete bkt;
                    }
                    
                    long already_found = 0;
                    #if BOOST_AMX_SIEVE
                    for (long i = 0; i < num_threads; i++) {
                        already_found += sol_list[i]->num_a_insert + sol_list[i]->num_s_insert;
                    }
                    #else
                    already_found = local_profile.succ_add2;
                    #endif
                    if (already_found > num_empty + sorted_index - goal_index) {
                        pthread_spin_lock(&local_profile.profile_lock);
                        if (too_many_sol) {
                            pthread_spin_unlock(&local_profile.profile_lock);
                            break;
                        }
                        too_many_sol = 1;
                        long rb0_not_used = 0;
                        long rb1_not_used = 0;
                        long rb2_not_used = 0;
                        for (long i = 0; i < BGJ3_AMX_BUCKET0_BATCHSIZE; i++) {
                            if (rbucket1_nrem[i] > 0) {
                                rb1_not_used += rbucket1_nrem[i];
                                rb2_not_used += rbucket1_nrem[i];
                            }
                            if (rbucket1_nrem[i] == -1) rb0_not_used += 1;
                        }
                        if (rbucket0_nrem > 0) rb2_not_used = rbucket0_nrem;
                        local_profile.report_bucket_not_used(3, rb0_not_used, rb1_not_used, rb2_not_used);
                        pthread_spin_unlock(&local_profile.profile_lock);
                        break;
                    }
                }
                booster->filter_sol_list(sol_list[thread], goal_score);
            }

            #pragma omp parallel for
            for (long i = 0; i < num_threads * BGJ3_AMX_BUCKET2_BATCHSIZE; i++) {
                if (bucket2[i]) delete bucket2[i];
            }
            #pragma omp parallel for schedule (guided)
            for (long i = 0; i < BGJ3_AMX_BUCKET0_BATCHSIZE * BGJ3_AMX_BUCKET1_BATCHSIZE; i++) {
                if (bucket1[i]) delete bucket1[i];
                if (rbucket1[i]) delete rbucket1[i];
            }
            #pragma omp parallel for schedule (guided)
            for (long i = 0; i < BGJ3_AMX_BUCKET0_BATCHSIZE; i++) {
                if (rbucket0[i]) delete rbucket0[i];
            }
            #endif

            // check if we get stucked or finished
            do {
                num_total_sol = 0;
                for (long i = 0; i < num_threads; i++){
                    #if BOOST_AMX_SIEVE
                    num_total_sol += sol_list[i]->num_a_insert + sol_list[i]->num_s_insert;
                    #else
                    num_total_sol += sol_list[i]->num_sol();
                    #endif
                }
                if (first_collect_sol == -1.0) first_collect_sol = num_total_sol + 0.0;
                if (num_total_sol - last_num_total_sol <= (1 + stucktime) * first_collect_sol * 0.001){
                    stucktime++;
                } else {
                    stucktime = 0;
                    last_num_total_sol = num_total_sol;
                }
                if (num_total_sol > one_epoch_ratio * num_vec) rel_collection_stop = true;
                if (stucktime > AMX_MAX_STUCK_TIME) {
                    sieving_stucked = 1;
                    rel_collection_stop = true;
                } 
            } while (0);
        } while (!rel_collection_stop);

        #if BOOST_AMX_SIEVE
        for (long i = 0; i < num_threads; i++) {
            local_profile.filter_time += sol_list[i]->filter_time;
            local_profile.succ_insert += sol_list[i]->num_a_insert + sol_list[i]->num_s_insert;
            sol_list[i]->filter_time = 0.0;
        }
        local_profile.filter_time /= num_threads;
        #endif

        // only bucket1 and bucket0 are done parallelly
        #if AMX_PARALLEL_BUCKET1 == 0
        local_profile.bucket1_time /= num_threads;
        #endif
        local_profile.bucket2_time /= num_threads;
        local_profile.search0_time /= num_threads;
        local_profile.search1_time /= num_threads;
        local_profile.search2_time /= num_threads;
        local_profile.one_epoch_log(3);
        local_profile.sol_check(sol_list, num_threads, goal_norm, uid);
        main_profile.combine(&local_profile);

        ///////////////// inserting /////////////////
        TIMER_START;
        uint64_t num_total_insert;
        if (log_level >= 3) {
            num_total_insert = _pool_insert_amx<1>(sol_list, num_threads, goal_norm, goal_index, &main_profile);
        } else {
            num_total_insert = _pool_insert_amx<0>(sol_list, num_threads, goal_norm, goal_index, &main_profile);
        }
        TIMER_END;
        main_profile.insert_time += CURRENT_TIME;
        main_profile.succ_insert += num_total_insert;
        main_profile.insert_log(num_total_insert, CURRENT_TIME);

        if (log_level >= 4) check_pool_status(0, 1);
        
        if (resort_ratio * num_vec > sorted_index){
            TIMER_START;
            sort_cvec();
            TIMER_END;
            main_profile.sort_time += CURRENT_TIME;
        }

        for (long i = 0; i < num_threads; i++) delete sol_list[i];

        if (main_profile.num_epoch == max_epoch) break;

        do {
            std::ifstream VINstream(".vin");
            double sr = 0.0;
            VINstream >> sr;
            if (sr > 0.01 && sr < 1.0) saturation_ratio = sr;
            VINstream.close();
        } while (0);
    }

    main_profile.final_log(3, sieving_stucked);
    if (sieving_stucked || CSD == 80) {
        if (check_dim_lose()) {
            sieving_stucked = 0;
        } else {
            sieving_stucked = 1;
        }
    }
    return sieving_stucked;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::bgj_amx_upsieve(long log_level, long max_epoch, double goal_norm_scale, long ESD, double prefer_deep) {
    if (CSD < 94) {
        return bgj1_Sieve(log_level, 1);
    } else if (CSD < 104) {
        return bgj2_Sieve(log_level, 1);
    } else {
        return bgj3_Sieve_amx(log_level, max_epoch, goal_norm_scale, ESD, prefer_deep);
    }
}

template <uint32_t nb>
int Pool_epi8_t<nb>::bgj_amx_downsieve(long log_level, long max_epoch, double goal_norm_scale, long ESD, double prefer_deep) {
    return bgj_amx_upsieve(log_level, max_epoch, goal_norm_scale, ESD, prefer_deep);
}

template <uint32_t nb>
int Pool_epi8_t<nb>::left_progressive_amx(long ind_l, long ind_r, long num_threads, long log_level, long ssd) {
    if (ind_r - ind_l < ssd) {
        fprintf(stderr, "[Error] Pool_epi8_t<%u>::left_progressive_amx: sieving dim too small, aborted.\n", nb);
        return 0;
    }
    if (ind_r - ind_l > nb * 32) {
        fprintf(stderr, "[Error] Pool_epi8_t<%u>::left_progressive_amx: sieving dim(%ld) > nb(%u) * 32, aborted.\n", nb, ind_r - ind_l, nb);
        return -1;
    }

    int show_lift = 0;
    if (abs(log_level - 16384) < 10) {
        log_level -= 16384;
        show_lift = 1;
    }
    
    clear_pool();
    set_num_threads(num_threads);
    set_max_pool_size((long)(pow(4./3., (ind_r - ind_l) *0.5) * 3.2) + 1);
    set_sieving_context(ind_r - ssd, ind_r);
    sampling((pow(4./3., ssd * 0.5) * 6.0) > _pool_size ? _pool_size : (long)(pow(4./3., ssd * 0.5) * 6.0));
    bgj1_Sieve(log_level, 1);
    long msd = ind_r - ind_l;
    while(index_l > ind_l) {
        extend_left();
        long target_num_vec = (long) (pow(4./3., CSD * 0.5) * 3.2);
        if (target_num_vec > num_vec + num_empty) num_empty = target_num_vec - num_vec;
        bgj_amx_upsieve(log_level, -1, -1.0, -1, CSD <= msd - 8 ? -1.0 : -1.0 - AMX_BOOST_DOWNSIEVE_MASK_RATIO * (-msd + CSD + 8));
        if (show_lift) show_min_lift(0);
    }
    return 1;
}


template struct bgj_amx_profile_data_t<5>;
template int Pool_epi8_t<5>::bgj3_Sieve_amx(long log_level, long max_epoch, double goal_norm_scale, long ESD, double prefer_deep);
template int Pool_epi8_t<5>::bgj_amx_upsieve(long log_level, long max_epoch, double goal_norm_scale, long ESD, double prefer_deep);
template int Pool_epi8_t<5>::bgj_amx_downsieve(long log_level, long max_epoch, double goal_norm_scale, long ESD, double prefer_deep);
template int Pool_epi8_t<5>::left_progressive_amx(long ind_l, long ind_r, long num_threads, long log_level, long ssd);
#endif