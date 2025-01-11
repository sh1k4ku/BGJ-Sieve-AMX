#include "../include/bucket_epi8.h"
#include "../include/bgj_epi8.h"


///////////////// bucket_epi8_t /////////////////

template <bool record_dp>
int bucket_epi8_t<record_dp>::_clear() {
    if (pvec) free(pvec);
    if (nvec) free(nvec);
    if (record_dp && pdot) free(pdot);
    if (record_dp && ndot) free(ndot);
    if (psum) free(psum);
    if (nsum) free(nsum);
    if (pnorm) free(pnorm);
    if (nnorm) free(nnorm);
    pvec = NULL; nvec = NULL;
    if (record_dp) { pdot = NULL; ndot = NULL; }
    psum = NULL; nsum = NULL;
    pnorm = NULL; nnorm = NULL;
    return 0;
}

template <bool record_dp>
int bucket_epi8_t<record_dp>::_alloc(long size, bool p) {
    if (p) {
        if (size <= _psize) return 0;
        pvec = (uint32_t *) realloc(pvec, size * sizeof(uint32_t));
        pnorm = (int32_t *) realloc(pnorm, size * sizeof(int32_t));
        psum = (int32_t *) realloc(psum, size * sizeof(int32_t));
        if (record_dp) pdot = (int32_t *) realloc(pdot, size * sizeof(int32_t));
        _psize = size;
        return (pnorm && pvec && psum && (!record_dp || pdot));
    } else {
        if (size <= _nsize) return 0;
        nvec = (uint32_t *) realloc(nvec, size * sizeof(uint32_t));
        nnorm = (int32_t *) realloc(nnorm, size * sizeof(int32_t));
        nsum = (int32_t *) realloc(nsum, size * sizeof(int32_t));
        if (record_dp) ndot = (int32_t *) realloc(ndot, size * sizeof(int32_t));
        _nsize = size;
        return (nnorm && nvec && nsum && (!record_dp || ndot));
    }
}

template <bool record_dp>
void bucket_epi8_t<record_dp>::combine(bucket_epi8_t<record_dp> **subbucket_list, long len) {
    if (num_pvec || num_nvec) {
        fprintf(stderr, "[Error] bucket_epi8_t<%d>::combine: combining subbuckets to nonempty buckets, aborted.\n", record_dp ? 1 : 0);
        return;
    }
    long total_psize = 0, total_nsize = 0;
    for (long i = 0; i < len; i++){
        total_psize += subbucket_list[i]->num_pvec;
        total_nsize += subbucket_list[i]->num_nvec;   
    }
    _alloc(total_psize, 1);
    _alloc(total_nsize, 0);
    for (long i = 0; i < len; i++){
        memcpy(pvec + num_pvec, subbucket_list[i]->pvec, subbucket_list[i]->num_pvec * sizeof(uint32_t));
        memcpy(nvec + num_nvec, subbucket_list[i]->nvec, subbucket_list[i]->num_nvec * sizeof(uint32_t));
        memcpy(pnorm + num_pvec, subbucket_list[i]->pnorm, subbucket_list[i]->num_pvec * sizeof(int32_t));
        memcpy(nnorm + num_nvec, subbucket_list[i]->nnorm, subbucket_list[i]->num_nvec * sizeof(int32_t));
        memcpy(psum + num_pvec, subbucket_list[i]->psum, subbucket_list[i]->num_pvec * sizeof(int32_t));
        memcpy(nsum + num_nvec, subbucket_list[i]->nsum, subbucket_list[i]->num_nvec * sizeof(int32_t));
        if (record_dp) memcpy(pdot + num_pvec, subbucket_list[i]->pdot, subbucket_list[i]->num_pvec * sizeof(int32_t));
        if (record_dp) memcpy(ndot + num_nvec, subbucket_list[i]->ndot, subbucket_list[i]->num_nvec * sizeof(int32_t));
        num_pvec += subbucket_list[i]->num_pvec;
        num_nvec += subbucket_list[i]->num_nvec;
    }
}

template <bool record_dp>
int bucket_epi8_t<record_dp>::remove_center(int max_unordered) {
    if (pvec == NULL) return 0;
    long low = 0;
    long high = num_pvec - 1;
    long target_ind = -1;
    do {
        while (high - low > 1) {
            long mid = (high + low) / 2; 
            if (pvec[mid] >= center_ind) {
                high = mid;
            } else {
                low = mid;
            }
        }
        if (pvec[low] == center_ind) {
            target_ind = low;
            break;
        }
        if (pvec[high] == center_ind) {
            target_ind = high;
            break;
        }
        // not find ?
        if (target_ind == -1) {
            if (pvec[low] >= pvec[num_pvec - 1]) {
                if (pvec[high] >= pvec[num_pvec - 1]) {
                    if (low > 0) {
                        if (pvec[low-1] >= center_ind) {
                            high = low - 1;
                            low = 0;
                        } else if (high < num_pvec - 1) {
                            low = high + 1;
                            high = num_pvec - 1;
                        }
                    } else if (num_pvec > 1) {
                        low = 1;
                        high = num_pvec - 1;
                    }
                } else {
                    if (pvec[high] < center_ind) {
                        low = high;
                        high = num_pvec - 1;
                    } else if (low > 0) {
                        high = low - 1;
                        low = 0;
                    }
                }
            } else if (pvec[high] >= pvec[num_pvec - 1]) {
                if (high < num_pvec - 1) {
                    low = high + 1;
                    high = num_pvec - 1;
                }
            } else if (pvec[num_pvec - 1] == center_ind) {
                target_ind = num_pvec - 1;
                break;
            } else {
                low = 0;
                high = num_pvec - 2;
                return -1;
            }
        }
    } while (max_unordered -->= 0);
    if (target_ind == -1) {
        for (long i = 0; i < num_pvec; i++) {
            if (pvec[i] == center_ind) {
                target_ind = i;
                break;
            }
        }
        if (target_ind == -1) {
            fprintf(stderr, "# center not found in the bucket\n");
            return -1;
        } else {
            if (num_pvec >= 2) {
                if (pvec[target_ind] < pvec[num_pvec - 2] && pvec[target_ind] < pvec[num_pvec - 1]) {
                    fprintf(stderr, "# center finally found by a brute force search\n");
                }
            }
        }
    } 
    
    num_pvec--;
    pvec[target_ind] = pvec[num_pvec];
    pnorm[target_ind] = pnorm[num_pvec];
    psum[target_ind] = psum[num_pvec];
    if (record_dp) pdot[target_ind] = pdot[num_pvec];
    
    return 1;
}

template class bucket_epi8_t<1>;
template class bucket_epi8_t<0>;


///////////////// sol_list_epi8_t /////////////////

int sol_list_epi8_t::_clear(){
    if (a_list) free(a_list);           // 0
    if (s_list) free(s_list);           // 1
    if (aa_list) free(aa_list);         // 2
    if (sa_list) free(sa_list);         // 3
    if (ss_list) free(ss_list);         // 4
    a_list = NULL; s_list = NULL;
    aa_list = NULL; sa_list = NULL; ss_list = NULL;
    return 0;
}

int sol_list_epi8_t::_alloc(long size, long type) {
    #define _REALLOC_PTR(__ptr, __orgsize, __r) do {                        \
        if (size <= __orgsize) return 0;                                    \
        __ptr = (uint32_t *) realloc(__ptr, size * sizeof(uint32_t) * __r); \
        __orgsize = size;                                                   \
        return (__ptr != NULL);                                             \
    } while (0);

    if (type == 0) _REALLOC_PTR(a_list, _asize, 2);
    if (type == 1) _REALLOC_PTR(s_list, _ssize, 2);
    if (type == 2) _REALLOC_PTR(aa_list, _aasize, 3);
    if (type == 3) _REALLOC_PTR(sa_list, _sasize, 3);
    if (type == 4) _REALLOC_PTR(ss_list, _sssize, 3);
    #undef _REALLOC_PTR
    return -1;
}

long sol_list_epi8_t::num_sol() {
    return num_a + num_s + num_aa + num_sa + num_ss;
}


///////////////// bgjn sieve operations /////////////////

template <uint32_t nb>
template <uint32_t batchsize, bool record_dp, bool faraway_center, bool for_bgj1, bool init_sieve>
int Pool_epi8_t<nb>::_pool_bucketing(bucket_epi8_t<record_dp> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2) {
    /////// prepare dst and local buffer ///////
    for (long i = 0; i < batchsize; i++) {
        dst3[i] = new bucket_epi8_t<record_dp>;
        if (!for_bgj1) dst2[i] = new bucket_epi8_t<0>;
    }
    bucket_epi8_t<record_dp> *local_bucket3[MAX_NTHREADS];
    bucket_epi8_t<0> *local_bucket2[MAX_NTHREADS];
    const uint16_t alpha3x2_epu16 = round(65536.0 * 2.0 * alpha3);
    const uint16_t alpha2x2_epu16 = round(65536.0 * 2.0 * alpha2);
    const __m256i alpha3x2_si256 = _mm256_set1_epi32(alpha3x2_epu16);
    const __m256i alpha2x2_si256 = _mm256_set1_epi32(alpha2x2_epu16);
    const long expect_bucket3_size = pow(1.0 - alpha3 * alpha3, CSD * 0.5) * num_vec;
    const long expect_bucket2_size = pow(1.0 - alpha2 * alpha2, CSD * 0.5) * num_vec;
    
    /////// choose centers ///////
    uint32_t center_ind_list[batchsize];
    uint8_t *center = (uint8_t *) NEW_VEC(batchsize * vec_length, sizeof(uint8_t));
    long num_try_find_center = 0;
    for (long i = 0; i < batchsize; i++) {
        int pass;
        do {
            pass = 1;
            center_ind_list[i] = rand() % ((long)(0.65 * num_vec));     // it really matters?
            if (num_try_find_center < 2000) {
                for (long j = 0; j < i; j++){
                    if (center_ind_list[j] == center_ind_list[i]) pass = 0;
                }
            }
            int8_t *ptr = vec + center_ind_list[i] * vec_length;
            num_try_find_center++;
            if (faraway_center) {
                if ((CSD * (int)ptr[0] * (int)ptr[0] > 2 * vnorm[center_ind_list[i]]) && (num_try_find_center < 1000)) pass = 0;
            }
            dst3[i]->set_center(center_ind_list[i], vnorm[center_ind_list[i]], vu[center_ind_list[i]]);
            if (!for_bgj1) dst2[i]->set_center(center_ind_list[i], vnorm[center_ind_list[i]], vu[center_ind_list[i]]);
            for (long j = 0; j < nb; j++) {
                __m256i x = _mm256_load_si256((__m256i *)(vec + center_ind_list[i] * vec_length + j * 32));
                _mm256_store_si256((__m256i *)(center + i * vec_length + j * 32), _mm256_xor_si256(epi8_sign_bit, x));
            }
        } while(!pass);
    }

    /////// each thread collect vectors in local buckets ///////
    #pragma omp parallel for 
    for (long thread = 0; thread < num_threads; thread++) {
        ///// prepare local buckets //////
        local_bucket3[thread] = new bucket_epi8_t<record_dp>[batchsize];
        if (!for_bgj1) local_bucket2[thread] = new bucket_epi8_t<0>[batchsize];
        for (long i = 0; i < batchsize; i++){
            local_bucket3[thread][i]._alloc(0.5 * expect_bucket3_size/num_threads, 1);
            local_bucket3[thread][i]._alloc(0.5 * expect_bucket3_size/num_threads, 0);
            if (for_bgj1) continue;
            local_bucket2[thread][i]._alloc(0.5 * expect_bucket2_size/num_threads, 1);
            local_bucket2[thread][i]._alloc(0.5 * expect_bucket2_size/num_threads, 0);
        }

        ///// prepare bucketing buffer /////
        #if PBUCKET_USE_BUFFER
        constexpr uint32_t csndp_ele_size = (init_sieve ? 12 : 10) + (record_dp ? 4 : 0);
        constexpr uint32_t csnp_ele_size = init_sieve ? 12 : 10;
        uint16_t *csndp_buffer, *csndp_pbuffer, *csndp_nbuffer;     // for bucket3, 16 or less Bytes for each vector
        uint16_t *csnp_pbuffer, *csnp_nbuffer;                      // for bucket2, 12 or less Bytes for each vector
        if (record_dp) csndp_buffer = (uint16_t *) malloc(PBUCKET_BUFFER_INIT_SIZE * csndp_ele_size);
        if (!record_dp) csndp_pbuffer = (uint16_t *) malloc((PBUCKET_BUFFER_INIT_SIZE >> 1) * csndp_ele_size);
        if (!record_dp) csndp_nbuffer = (uint16_t *) malloc((PBUCKET_BUFFER_INIT_SIZE >> 1) * csndp_ele_size);
        if (!for_bgj1) csnp_pbuffer = (uint16_t *) malloc((PBUCKET_BUFFER_INIT_SIZE >> 1) * csnp_ele_size); 
        if (!for_bgj1) csnp_nbuffer = (uint16_t *) malloc((PBUCKET_BUFFER_INIT_SIZE >> 1) * csnp_ele_size);
        long _csndp_buffer_size = PBUCKET_BUFFER_INIT_SIZE;
        long _csndp_buffer_psize = PBUCKET_BUFFER_INIT_SIZE >> 1;
        long _csndp_buffer_nsize = PBUCKET_BUFFER_INIT_SIZE >> 1;
        long _csnp_buffer_psize = PBUCKET_BUFFER_INIT_SIZE >> 1;
        long _csnp_buffer_nsize = PBUCKET_BUFFER_INIT_SIZE >> 1;
        long num_csndp = 0, num_pcsndp = 0, num_ncsndp = 0, num_pcsnp = 0, num_ncsnp = 0;
        #endif

        /** 
         * \param[in] __cind    uint32_t
         * \param[in] __sum     int32_t
         * \param[in] __norm    uint32_t
         * \param[in] __dp      int32_t
         * \param[in] __ind     long 
        */
        #define _ADD_TO_CSNDP_BUFFER(__cind, __sum, __norm, __dp, __ind)                                \
                                                                         do {                           \
            if (record_dp) {                                                                            \
                csndp_buffer[num_csndp*(csndp_ele_size>>1)] = __cind;                                   \
                *((int16_t *)(&csndp_buffer[num_csndp*(csndp_ele_size>>1)+1])) = (__sum) >> 7;          \
                if (init_sieve) {                                                                       \
                    *((uint32_t *)(&csndp_buffer[num_csndp*(csndp_ele_size>>1)+2])) = __norm;           \
                    *((int32_t *)(&csndp_buffer[num_csndp*(csndp_ele_size>>1)+4])) = __dp;              \
                    *((uint32_t *)(&csndp_buffer[num_csndp*(csndp_ele_size>>1)+6])) = __ind;            \
                } else {                                                                                \
                    csndp_buffer[num_csndp*(csndp_ele_size>>1)+2] = __norm;                             \
                    *((int32_t *)(&csndp_buffer[num_csndp*(csndp_ele_size>>1)+3])) = __dp;              \
                    *((uint32_t *)(&csndp_buffer[num_csndp*(csndp_ele_size>>1)+5])) = __ind;            \
                }                                                                                       \
                num_csndp++;                                                                            \
            } else {                                                                                    \
                if (init_sieve) {                                                                       \
                    if (__dp > 0) {                                                                     \
                        csndp_pbuffer[num_pcsndp*(csndp_ele_size>>1)] = __cind;                         \
                        *((int16_t *)(&csndp_pbuffer[num_pcsndp*(csndp_ele_size>>1)+1])) = (__sum) >> 7;\
                        *((uint32_t *)(&csndp_pbuffer[num_pcsndp*(csndp_ele_size>>1)+2])) = __norm;     \
                        *((uint32_t *)(&csndp_pbuffer[num_pcsndp*(csndp_ele_size>>1)+4])) = __ind;      \
                        num_pcsndp++;                                                                   \
                    } else {                                                                            \
                        csndp_nbuffer[num_ncsndp*(csndp_ele_size>>1)] = __cind;                         \
                        *((int16_t *)(&csndp_nbuffer[num_ncsndp*(csndp_ele_size>>1)+1])) = (__sum) >> 7;\
                        *((uint32_t *)(&csndp_nbuffer[num_ncsndp*(csndp_ele_size>>1)+2])) = __norm;     \
                        *((uint32_t *)(&csndp_nbuffer[num_ncsndp*(csndp_ele_size>>1)+4])) = __ind;      \
                        num_ncsndp++;                                                                   \
                    }                                                                                   \
                } else {                                                                                \
                    if (__dp > 0) {                                                                     \
                        csndp_pbuffer[num_pcsndp*(csndp_ele_size>>1)] = __cind;                         \
                        *((int16_t *)(&csndp_pbuffer[num_pcsndp*(csndp_ele_size>>1)+1])) = (__sum) >> 7;\
                        csndp_pbuffer[num_pcsndp*(csndp_ele_size>>1)+2] = __norm;                       \
                        *((uint32_t *)(&csndp_pbuffer[num_pcsndp*(csndp_ele_size>>1)+3])) = __ind;      \
                        num_pcsndp++;                                                                   \
                    } else {                                                                            \
                        csndp_nbuffer[num_ncsndp*(csndp_ele_size>>1)] = __cind;                         \
                        *((int16_t *)(&csndp_nbuffer[num_ncsndp*(csndp_ele_size>>1)+1])) = (__sum) >> 7;\
                        csndp_nbuffer[num_ncsndp*(csndp_ele_size>>1)+2] = __norm;                       \
                        *((uint32_t *)(&csndp_nbuffer[num_ncsndp*(csndp_ele_size>>1)+3])) = __ind;      \
                        num_ncsndp++;                                                                   \
                    }                                                                                   \
                }                                                                                       \
            }                                                                                           \
        } while (0)
        #define _ADD_TO_CSNP_BUFFER(__cind, __sum, __norm, __dp, __ind)                         \
                                                                  do {                          \
            if (__dp > 0) {                                                                     \
                csnp_pbuffer[num_pcsnp*(csnp_ele_size>>1)] = __cind;                            \
                *((int16_t *)(&csnp_pbuffer[num_pcsnp*(csnp_ele_size>>1)+1])) = (__sum) >> 7;   \
                if (init_sieve) {                                                               \
                    *((int32_t *)(&csnp_pbuffer[num_pcsnp*(csnp_ele_size>>1)+2])) = __norm;     \
                    *((uint32_t *)(&csnp_pbuffer[num_pcsnp*(csnp_ele_size>>1)+4])) = __ind;     \
                } else {                                                                        \
                    csnp_pbuffer[num_pcsnp*(csnp_ele_size>>1)+2] = __norm;                      \
                    *((uint32_t *)(&csnp_pbuffer[num_pcsnp*(csnp_ele_size>>1)+3])) = __ind;     \
                }                                                                               \
                num_pcsnp++;                                                                    \
            } else {                                                                            \
                csnp_nbuffer[num_ncsnp*(csnp_ele_size>>1)] = __cind;                            \
                *((int16_t *)(&csnp_nbuffer[num_ncsnp*(csnp_ele_size>>1)+1])) = (__sum) >> 7;   \
                if (init_sieve) {                                                               \
                    *((int32_t *)(&csnp_nbuffer[num_ncsnp*(csnp_ele_size>>1)+2])) = __norm;     \
                    *((uint32_t *)(&csnp_nbuffer[num_ncsnp*(csnp_ele_size>>1)+4])) = __ind;     \
                } else {                                                                        \
                    csnp_nbuffer[num_ncsnp*(csnp_ele_size>>1)+2] = __norm;                      \
                    *((uint32_t *)(&csnp_nbuffer[num_ncsnp*(csnp_ele_size>>1)+3])) = __ind;     \
                }                                                                               \
                num_ncsnp++;                                                                    \
            }                                                                                   \
        } while (0)
        #define ADD_TO_LOCAL_BUFFER(__cind, __sum, __norm, __dp, __ind, __3bound)               \
                                                                                  do {          \
            if (for_bgj1) {                                                                     \
                _ADD_TO_CSNDP_BUFFER(__cind, __sum, __norm, __dp, __ind);                       \
            } else {                                                                            \
                _ADD_TO_CSNP_BUFFER(__cind, __sum, __norm, __dp, __ind);                        \
                if (abs(__dp) > __3bound) {                                                     \
                    _ADD_TO_CSNDP_BUFFER(__cind, __sum, __norm, __dp, __ind);                   \
                }                                                                               \
            }                                                                                   \
        } while (0)
        #define PROCESS_CSNDP_BUFFER                                                                    \
                                     do {                                                               \
            if (record_dp) {                                                                            \
                for (long i = 0; i < num_csndp; i++) {                                                  \
                    if (init_sieve) {                                                                   \
                        local_bucket3[thread][csndp_buffer[i*(csndp_ele_size>>1)]].add_vec(             \
                            *((uint32_t *)(&csndp_buffer[i*(csndp_ele_size>>1)+6])),                    \
                            *((uint32_t *)(&csndp_buffer[i*(csndp_ele_size>>1)+2])),                    \
                            ((int32_t) (*((int16_t *)(&csndp_buffer[i*(csndp_ele_size>>1)+1])))) << 7,  \
                            *((int32_t *)(&csndp_buffer[i*(csndp_ele_size>>1)+4]))                      \
                        );                                                                              \
                    } else {                                                                            \
                        local_bucket3[thread][csndp_buffer[i*(csndp_ele_size>>1)]].add_vec(             \
                            *((uint32_t *)(&csndp_buffer[i*(csndp_ele_size>>1)+5])),                    \
                            csndp_buffer[i*(csndp_ele_size>>1)+2],                                      \
                            ((int32_t) (*((int16_t *)(&csndp_buffer[i*(csndp_ele_size>>1)+1])))) << 7,  \
                            *((int32_t *)(&csndp_buffer[i*(csndp_ele_size>>1)+3]))                      \
                        );                                                                              \
                    }                                                                                   \
                }                                                                                       \
                num_csndp = 0;                                                                          \
            } else {                                                                                    \
                for (long i = 0; i < num_pcsndp; i++) {                                                 \
                    if (init_sieve) {                                                                   \
                        local_bucket3[thread][csndp_pbuffer[i*(csndp_ele_size>>1)]].add_pvec(           \
                            *((uint32_t *)(&csndp_pbuffer[i*(csndp_ele_size>>1)+4])),                   \
                            *((uint32_t *)(&csndp_pbuffer[i*(csndp_ele_size>>1)+2])),                   \
                            ((int32_t) (*((int16_t *)(&csndp_pbuffer[i*(csndp_ele_size>>1)+1])))) << 7  \
                        );                                                                              \
                    } else {                                                                            \
                        local_bucket3[thread][csndp_pbuffer[i*(csndp_ele_size>>1)]].add_pvec(           \
                            *((uint32_t *)(&csndp_pbuffer[i*(csndp_ele_size>>1)+3])),                   \
                            csndp_pbuffer[i*(csndp_ele_size>>1)+2],                                     \
                            ((int32_t) (*((int16_t *)(&csndp_pbuffer[i*(csndp_ele_size>>1)+1])))) << 7  \
                        );                                                                              \
                    }                                                                                   \
                }                                                                                       \
                for (long i = 0; i < num_ncsndp; i++) {                                                 \
                    if (init_sieve) {                                                                   \
                        local_bucket3[thread][csndp_nbuffer[i*(csndp_ele_size>>1)]].add_nvec(           \
                            *((uint32_t *)(&csndp_nbuffer[i*(csndp_ele_size>>1)+4])),                   \
                            *((uint32_t *)(&csndp_nbuffer[i*(csndp_ele_size>>1)+2])),                   \
                            ((int32_t) (*((int16_t *)(&csndp_nbuffer[i*(csndp_ele_size>>1)+1])))) << 7  \
                        );                                                                              \
                    } else {                                                                            \
                        local_bucket3[thread][csndp_nbuffer[i*(csndp_ele_size>>1)]].add_nvec(           \
                            *((uint32_t *)(&csndp_nbuffer[i*(csndp_ele_size>>1)+3])),                   \
                            csndp_nbuffer[i*(csndp_ele_size>>1)+2],                                     \
                            ((int32_t) (*((int16_t *)(&csndp_nbuffer[i*(csndp_ele_size>>1)+1])))) << 7  \
                        );                                                                              \
                    }                                                                                   \
                }                                                                                       \
                num_pcsndp = 0;                                                                         \
                num_ncsndp = 0;                                                                         \
            }                                                                                           \
        } while (0)
        #define PROCESS_CSNP_BUFFER                                                                 \
                                    do {                                                            \
            for (long i = 0; i < num_pcsnp; i++) {                                                  \
                if (init_sieve) {                                                                   \
                    local_bucket2[thread][csnp_pbuffer[i*(csnp_ele_size>>1)]].add_pvec(             \
                        *((uint32_t *)(&csnp_pbuffer[i*(csnp_ele_size>>1)+4])),                     \
                        *((uint32_t *)(&csnp_pbuffer[i*(csnp_ele_size>>1)+2])),                     \
                        ((int32_t) (*((int16_t *)(&csnp_pbuffer[i*(csnp_ele_size>>1)+1])))) << 7    \
                    );                                                                              \
                } else {                                                                            \
                    local_bucket2[thread][csnp_pbuffer[i*(csnp_ele_size>>1)]].add_pvec(             \
                        *((uint32_t *)(&csnp_pbuffer[i*(csnp_ele_size>>1)+3])),                     \
                        csnp_pbuffer[i*(csnp_ele_size>>1)+2],                                       \
                        ((int32_t) (*((int16_t *)(&csnp_pbuffer[i*(csnp_ele_size>>1)+1])))) << 7    \
                    );                                                                              \
                }                                                                                   \
            }                                                                                       \
            for (long i = 0; i < num_ncsnp; i++) {                                                  \
                if (init_sieve) {                                                                   \
                    local_bucket2[thread][csnp_nbuffer[i*(csnp_ele_size>>1)]].add_nvec(             \
                        *((uint32_t *)(&csnp_nbuffer[i*(csnp_ele_size>>1)+4])),                     \
                        *((uint32_t *)(&csnp_nbuffer[i*(csnp_ele_size>>1)+2])),                     \
                        ((int32_t) (*((int16_t *)(&csnp_nbuffer[i*(csnp_ele_size>>1)+1])))) << 7    \
                    );                                                                              \
                } else {                                                                            \
                    local_bucket2[thread][csnp_nbuffer[i*(csnp_ele_size>>1)]].add_nvec(             \
                        *((uint32_t *)(&csnp_nbuffer[i*(csnp_ele_size>>1)+3])),                     \
                        csnp_nbuffer[i*(csnp_ele_size>>1)+2],                                       \
                        ((int32_t) (*((int16_t *)(&csnp_nbuffer[i*(csnp_ele_size>>1)+1])))) << 7    \
                    );                                                                              \
                }                                                                                   \
            }                                                                                       \
            num_pcsnp = 0;                                                                          \
            num_ncsnp = 0;                                                                          \
        } while (0)
        
        #define PROCESS_AND_FREE_LOCAL_BUFFER   \
                                     do {       \
            PROCESS_CSNDP_BUFFER;               \
            if (!for_bgj1) PROCESS_CSNP_BUFFER; \
            if (record_dp) {                    \
                free(csndp_buffer);             \
            } else {                            \
                free(csndp_pbuffer);            \
                free(csndp_nbuffer);            \
            }                                   \
            if (!for_bgj1) free(csnp_pbuffer);  \
            if (!for_bgj1) free(csnp_nbuffer);  \
        } while(0)
        #define CHECK_BUFFER_FULL                                                                                       \
                                  do {                                                                                  \
            if (record_dp) {                                                                                            \
                if (num_csndp + 8 * batchsize > _csndp_buffer_size) {                                                   \
                    if (PBUCKET_BUFFER_SIZE_DYNAMIC) {                                                                  \
                        csndp_buffer = (uint16_t *) realloc(csndp_buffer, _csndp_buffer_size * 2 * csndp_ele_size);     \
                        _csndp_buffer_size *= 2;                                                                        \
                    } else {                                                                                            \
                        PROCESS_CSNDP_BUFFER;                                                                           \
                    }                                                                                                   \
                }                                                                                                       \
            } else {                                                                                                    \
                if ((num_pcsndp + 8 * batchsize > _csndp_buffer_psize) ||                                               \
                    (num_ncsndp + 8 * batchsize > _csndp_buffer_nsize)) {                                               \
                    if (PBUCKET_BUFFER_SIZE_DYNAMIC) {                                                                  \
                        csndp_pbuffer = (uint16_t *) realloc(csndp_pbuffer, _csndp_buffer_psize * 2 * csndp_ele_size);  \
                        csndp_nbuffer = (uint16_t *) realloc(csndp_nbuffer, _csndp_buffer_nsize * 2 * csndp_ele_size);  \
                        _csndp_buffer_psize *= 2;                                                                       \
                        _csndp_buffer_nsize *= 2;                                                                       \
                    } else {                                                                                            \
                        PROCESS_CSNDP_BUFFER;                                                                           \
                    }                                                                                                   \
                }                                                                                                       \
            }                                                                                                           \
                                                                                                                        \
            if (!for_bgj1) {                                                                                            \
                if ((num_pcsnp + 8 * batchsize > _csnp_buffer_psize) ||                                                 \
                    (num_ncsnp + 8 * batchsize > _csnp_buffer_nsize)) {                                                 \
                    if (PBUCKET_BUFFER_SIZE_DYNAMIC) {                                                                  \
                        csnp_pbuffer = (uint16_t *) realloc(csnp_pbuffer, _csnp_buffer_psize * 2 * csnp_ele_size);      \
                        csnp_nbuffer = (uint16_t *) realloc(csnp_nbuffer, _csnp_buffer_nsize * 2 * csnp_ele_size);      \
                        _csnp_buffer_psize *= 2;                                                                        \
                        _csnp_buffer_nsize *= 2;                                                                        \
                    } else {                                                                                            \
                        PROCESS_CSNP_BUFFER;                                                                            \
                    }                                                                                                   \
                }                                                                                                       \
            }                                                                                                           \
        } while(0)
        

        #define ADD_TO_BUCKET(__cind, __sum, __norm, __dp, __ind, __3bound)                     \
                                                                                  do {          \
            if (for_bgj1) {                                                                     \
                if (record_dp) {                                                                \
                    local_bucket3[thread][__cind].add_vec(__ind, __norm, __sum, __dp);          \
                } else {                                                                        \
                    if (__dp > 0) {                                                             \
                        local_bucket3[thread][__cind].add_pvec(__ind, __norm, __sum);           \
                    } else {                                                                    \
                        local_bucket3[thread][__cind].add_nvec(__ind, __norm, __sum);           \
                    }                                                                           \
                }                                                                               \
            } else {                                                                            \
                if (__dp > 0) {                                                                 \
                    local_bucket2[thread][__cind].add_pvec(__ind, __norm, __sum);               \
                    if (abs(__dp) > __3bound) {                                                 \
                        if (record_dp) {                                                        \
                            local_bucket3[thread][__cind].add_vec(__ind, __norm, __sum, __dp);  \
                        } else {                                                                \
                            local_bucket3[thread][__cind].add_pvec(__ind, __norm, __sum);       \
                        }                                                                       \
                    }                                                                           \
                } else {                                                                        \
                    local_bucket2[thread][__cind].add_nvec(__ind, __norm, __sum);               \
                    if (abs(__dp) > __3bound) {                                                 \
                        if (record_dp) {                                                        \
                            local_bucket3[thread][__cind].add_vec(__ind, __norm, __sum, __dp);  \
                        } else {                                                                \
                            local_bucket3[thread][__cind].add_nvec(__ind, __norm, __sum);       \
                        }                                                                       \
                    }                                                                           \
                }                                                                               \
            }                                                                                   \
        } while (0)

        ///// main loop //////
        const long begin_ind = (thread*num_vec)/num_threads;
        const long end_ind = (thread*num_vec+num_vec)/num_threads;
        long ind = begin_ind;
        while (ind < end_ind - 7) {
            __m256i sum_b8 = _mm256_loadu_si256((__m256i *)(vsum + ind));
            __m256i norm_b8 = _mm256_loadu_si256((__m256i *)(vnorm + ind));
            __m256i bound3, bound2;
            int32_t *sum_epi32 = (int32_t *)(&sum_b8);
            uint32_t *norm_epi32 = (uint32_t *)(&norm_b8);
            int32_t *bound3_epi32 = (int32_t *)(&bound3);
            int32_t *bound2_epi32 = (int32_t *)(&bound2);
            if (init_sieve) {
                __m256i norm_hi = _mm256_srai_epi64(norm_b8, 32);
                __m256i bound3_lo = _mm256_srai_epi64(_mm256_mul_epi32(norm_b8, alpha3x2_si256), 16);
                __m256i bound3_hi = _mm256_slli_epi64(_mm256_mul_epi32(norm_hi, alpha3x2_si256), 16);
                if (!for_bgj1) {
                    __m256i bound2_lo = _mm256_srai_epi64(_mm256_mul_epi32(norm_b8, alpha2x2_si256), 16);
                    __m256i bound2_hi = _mm256_slli_epi64(_mm256_mul_epi32(norm_hi, alpha2x2_si256), 16);
                    bound2 = _mm256_blend_epi32(bound2_lo, bound2_hi, 170);
                }
                bound3 = _mm256_blend_epi32(bound3_lo, bound3_hi, 170);
            } else {
                bound3 = _mm256_mulhi_epu16(norm_b8, alpha3x2_si256);
                if (!for_bgj1) bound2 = _mm256_mulhi_epu16(norm_b8, alpha2x2_si256);
            }
            
            __m256i dst[batchsize];             // record the dot products
            int32_t *dst_epi32 = (int32_t *) (&dst[0]);
            uint64_t cdst[batchsize >> 3];      // record the cmp results
            uint32_t cmp[8];
            
            uint32_t cind = 0;
            while (cind < batchsize) {
                vdp8x8(&dst[cind], center + cind * vec_length, vec + ind * vec_length);
                for (long i = 0; i < 8; i++) dst[cind+i] = _mm256_sub_epi32(dst[cind+i], sum_b8);
                if (for_bgj1) {
                    for (long i = 0; i < 8; i++) cmp[i] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(_mm256_abs_epi32(dst[cind+i]), bound3));
                } else {
                    for (long i = 0; i < 8; i++) cmp[i] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(_mm256_abs_epi32(dst[cind+i]), bound2));
                }
                uint64_t cdst_lo = ( cmp[0] | (cmp[1] << 8) | (cmp[2] << 16) | (cmp[3] << 24) );
                uint64_t cdst_hi = ( cmp[4] | (cmp[5] << 8) | (cmp[6] << 16) | (cmp[7] << 24) );
                cdst[cind >> 3] = ( cdst_lo | (cdst_hi << 32) );
                cind += 8;
            }

            for (uint32_t i = 0; i < batchsize; i += 8) {
                while(cdst[i>>3]) {
                    uint32_t r = __builtin_ctzl(cdst[i>>3]);
                    cdst[i>>3] -= (1UL << r);
                    #if PBUCKET_USE_BUFFER
                    ADD_TO_LOCAL_BUFFER(i+(r>>3), sum_epi32[r&(0x7)], norm_epi32[r&(0x7)], dst_epi32[i*8+r], ind+(r&(0x7)), bound3_epi32[r&(0x7)]);
                    #else
                    ADD_TO_BUCKET(i+(r>>3), sum_epi32[r&(0x7)], norm_epi32[r&(0x7)], dst_epi32[i*8+r], ind+(r&(0x7)), bound3_epi32[r&(0x7)]);
                    #endif
                }
            }

            #if PBUCKET_USE_BUFFER
            CHECK_BUFFER_FULL;
            #endif

            ind += 8;
        }
        
        #if 1
        if (ind < end_ind) {
            const long nrem = end_ind - ind;

            __m256i sum_si256[7];
            __m256i bound2_si256[7];
            __m256i bound3_si256[7];
            int32_t *sum_epi32 = &vsum[ind];
            int32_t *norm_epi32 = &vnorm[ind];
            int32_t *bound3_epi32 = (int32_t *)&bound3_si256[0];
            for (long i = 0; i < nrem; i++) sum_si256[i] = _mm256_set1_epi32(sum_epi32[i]);
            for (long i = 0; i < nrem; i++) {
                bound3_si256[i] = _mm256_set1_epi32(((int64_t) norm_epi32[i] * (int64_t) alpha3x2_epu16) >> 16);
                if (!for_bgj1) bound2_si256[i] = _mm256_set1_epi32(((int64_t) norm_epi32[i] * (int64_t) alpha2x2_epu16) >> 16);
            }

            __m256i dst[7 * (batchsize>>3)];
            int32_t *dst_epi32 = (int32_t *) (&dst[0]);
            uint64_t cdst[batchsize >> 3];
            uint32_t cmp[8] = {};
            
            uint32_t cind = 0;
            while (cind < batchsize) {
                long bias = nrem * (cind >> 3);
                vdp8xn(&dst[bias], vec + ind * vec_length, center + cind * vec_length, nrem);
                for (long i = 0; i < nrem; i++) dst[bias+i] = _mm256_sub_epi32(dst[bias+i], sum_si256[i]);
                if (for_bgj1) {
                    for (long i = 0; i < nrem; i++) cmp[i] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(_mm256_abs_epi32(dst[bias+i]), bound3_si256[i]));
                } else {
                    for (long i = 0; i < nrem; i++) cmp[i] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(_mm256_abs_epi32(dst[bias+i]), bound2_si256[i]));
                }
                uint64_t cdst_lo = ( cmp[0] | (cmp[1] << 8) | (cmp[2] << 16) | (cmp[3] << 24) );
                if (nrem > 3) {
                    uint64_t cdst_hi = ( cmp[4] | (cmp[5] << 8) | (cmp[6] << 16) | (cmp[7] << 24) );
                    cdst[cind >> 3] = ( cdst_lo | (cdst_hi << 32) );
                } else {
                    cdst[cind >> 3] = cdst_lo;
                }
                
                cind += 8;
            }

            for (uint32_t i = 0; i < batchsize; i += 8) {
                while(cdst[i>>3]) {
                    uint32_t r = __builtin_ctzl(cdst[i>>3]);
                    cdst[i>>3] -= (1UL << r);
                    #if PBUCKET_USE_BUFFER
                    ADD_TO_LOCAL_BUFFER(i+(r & 0x7), sum_epi32[r>>3], norm_epi32[r >> 3], dst_epi32[i*nrem+r], ind+(r>>3), bound3_epi32[r]);
                    #else
                    ADD_TO_BUCKET(i+(r & 0x7), sum_epi32[r>>3], norm_epi32[r >> 3], dst_epi32[i*nrem+r], ind+(r>>3), bound3_epi32[r]);
                    #endif
                }
            }
        }
        
        #else
        while (ind < end_ind) {
            int32_t sum_epi32 = vsum[ind];
            int32_t norm_epi32 = vnorm[ind];
            int32_t bound3 = ((int64_t) norm_epi32 * (int64_t) alpha3x2_epu16) >> 16;
            int32_t bound2 = ((int64_t) norm_epi32 * (int64_t) alpha2x2_epu16) >> 16;
            __m256i sum_si256 = _mm256_set1_epi32(sum_epi32);
            __m256i bound3_si256 = _mm256_set1_epi32(bound3);
            __m256i bound2_si256 = _mm256_set1_epi32(bound2);


            __m256i dst[batchsize>>3];
            int32_t *dst_epi32 = (int32_t *) (&dst[0]);
            uint8_t cdst[batchsize>>3];

            uint32_t cind = 0;
            while (cind < batchsize) {
                dst[cind >> 3] = vdp8x1(vec+ind*vec_length, center+cind*vec_length);
                dst[cind >> 3] = _mm256_sub_epi32(dst[cind >> 3], sum_si256);
                if (for_bgj1) {
                    cdst[cind >> 3] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(_mm256_abs_epi32(dst[cind>>3]), bound3_si256));
                } else {
                    cdst[cind >> 3] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(_mm256_abs_epi32(dst[cind>>3]), bound2_si256));
                }
                cind += 8;
            }

            for (uint32_t i = 0; i < batchsize; i += 8) {
                while(cdst[i>>3]) {
                    uint32_t r = __builtin_ctzl(cdst[i>>3]);
                    cdst[i>>3] -= (1U << r);
                    #if PBUCKET_USE_BUFFER
                    ADD_TO_LOCAL_BUFFER(i+r, sum_epi32, norm_epi32, dst_epi32[i+r], ind, bound3);
                    #else
                    ADD_TO_BUCKET(i+r, sum_epi32, norm_epi32, dst_epi32[i+r], ind, bound3);
                    #endif
                }
            }

            ind++;
        }
        #endif

        #if PBUCKET_USE_BUFFER
        PROCESS_AND_FREE_LOCAL_BUFFER;
        #endif

        #undef _ADD_TO_CSNDP_BUFFER
        #undef _ADD_TO_CSNP_BUFFER
        #undef ADD_TO_LOCAL_BUFFER
        #undef PROCESS_CSNDP_BUFFER
        #undef PROCESS_CSNP_BUFFER
        #undef PROCESS_AND_FREE_LOCAL_BUFFER
        #undef CHECK_BUFFER_FULL
        #undef ADD_TO_BUCKET
    }

    /////// combine the data to main buckets ///////
    #pragma omp parallel for
    for (long i = 0; i < batchsize; i++) {
        bucket_epi8_t<record_dp> *tmp3[MAX_NTHREADS];
        bucket_epi8_t<0> *tmp2[MAX_NTHREADS];
        for (long j = 0; j < num_threads; j++) tmp3[j] = &local_bucket3[j][i];
        dst3[i]->combine(tmp3, num_threads);
        if (for_bgj1) continue;
        for (long j = 0; j < num_threads; j++) tmp2[j] = &local_bucket2[j][i];
        dst2[i]->combine(tmp2, num_threads);
    }

    /////// remove center in the buckets ///////
    #pragma omp parallel for
    for (long i = 0; i < batchsize; i++) {
        // our buckets are now well ordered
        if (record_dp) dst3[i]->remove_center(0);
        if (!for_bgj1) dst2[i]->remove_center(0);
        // then the output buckets may have one non-sorted element each
    }

    /////// free local buckets and center ///////
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++) {
        delete[] local_bucket3[thread];
        if (!for_bgj1) delete[] local_bucket2[thread];
    }
    FREE_VEC((void *) center);

    return 1;
}

#pragma region
#define TRY_ADDA_TO_DST(__src0, __src1)                     \
                                        do {                \
    try_add2++;                                             \
    uint64_t __u = vu[__src0] + vu[__src1];                 \
    if (uid->insert_uid(__u)) {                             \
        succ_add2++;                                        \
        sol->add_sol_a(__src0, __src1);                     \
    }                                                       \
} while (0)

#define TRY_ADDS_TO_DST(__src0, __src1)                     \
                                        do {                \
    try_add2++;                                             \
    uint64_t __u = vu[__src0] - vu[__src1];                 \
    if (uid->insert_uid(__u)) {                             \
        succ_add2++;                                        \
        sol->add_sol_s(__src0, __src1);                     \
    }                                                       \
} while (0)

#define TRY_ADDAA_TO_DST(__src0, __src1)                    \
                                        do {                \
    try_add3++;                                             \
    uint64_t __u = bkt->center_u + vu[__src0] + vu[__src1]; \
    if (uid->insert_uid(__u)) {                             \
        succ_add3++;                                        \
        sol->add_sol_aa(bkt->center_ind, __src0, __src1);   \
    }                                                       \
} while (0)

#define TRY_ADDSA_TO_DST(__src0, __src1)                    \
                                        do {                \
    try_add3++;                                             \
    uint64_t __u = bkt->center_u - vu[__src0] + vu[__src1]; \
    if (uid->insert_uid(__u)) {                             \
        succ_add3++;                                        \
        sol->add_sol_sa(bkt->center_ind, __src0, __src1);   \
    }                                                       \
} while (0)

#define TRY_ADDAS_TO_DST(__src0, __src1)                    \
                                        do {                \
    try_add3++;                                             \
    uint64_t __u = bkt->center_u + vu[__src0] - vu[__src1]; \
    if (uid->insert_uid(__u)) {                             \
        succ_add3++;                                        \
        sol->add_sol_sa(bkt->center_ind, __src1, __src0);   \
    }                                                       \
} while (0)

#define TRY_ADDSS_TO_DST(__src0, __src1)                    \
                                        do {                \
    try_add3++;                                             \
    uint64_t __u = bkt->center_u - vu[__src0] - vu[__src1]; \
    if (uid->insert_uid(__u)) {                             \
        succ_add3++;                                        \
        sol->add_sol_ss(bkt->center_ind, __src0, __src1);   \
    }                                                       \
} while (0)

// only used in _search_cred
#define CHECK_AND_ADD_CRED_1X64(__cmp, __ptr, __add_func)       \
                                                          do {  \
    while (__cmp) {                                             \
        int __r = __builtin_ctzl(__cmp);                        \
        __cmp -= (1UL << __r);                                  \
        __add_func(bkt->center_ind, (__ptr)[__r]);              \
    }                                                           \
} while(0)

// only used in _search_pp/np/nn
#define CHECK_AND_ADD_8x8(__cmp, __ptr1, __ptr2, __add_func)    \
                                                 do {           \
    while (__cmp) {                                             \
        int __r = __builtin_ctzl(__cmp);                        \
        __cmp -= (1UL << __r);                                  \
        __add_func(__ptr1[__r >> 3], __ptr2[__r & 0x7]);        \
    }                                                           \
} while (0)
#pragma endregion

template <uint32_t nb>
template <uint32_t batchsize, bool faraway_center, bool for_bgj2, bool init_sieve, bool profiling>
int Pool_epi8_t<nb>::_sub_bucketing(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<nb> *prof) {
    uint64_t try_add2 = 0;
    uint64_t try_add3 = 0;
    uint64_t succ_add2 = 0;
    uint64_t succ_add3 = 0;

    /////// prepare dst and local buffer ///////
    for (long i = 0; i < batchsize; i++) {
        if (!dst3[i]) {
            dst3[i] = new bucket_epi8_t<0>;
        } else {
            dst3[i]->num_pvec = 0;
            dst3[i]->num_nvec = 0;
        }
        if (!for_bgj2 && !dst2[i]) {
            dst2[i] = new bucket_epi8_t<0>;
        } else if (!for_bgj2) {
            dst2[i]->num_pvec = 0;
            dst2[i]->num_nvec = 0;
        }
    }
    const uint16_t alpha3x2_epu16 = round(65536.0 * 2.0 * alpha3);
    const uint16_t alpha2x2_epu16 = round(65536.0 * 2.0 * alpha2);
    const __m256i alpha3x2_si256 = _mm256_set1_epi32(alpha3x2_epu16);
    const __m256i alpha2x2_si256 = _mm256_set1_epi32(alpha2x2_epu16);
    const long expect_bucket3_size = pow(1.0 - alpha3 * alpha3, CSD * 0.5) * (main_bucket->num_pvec + main_bucket->num_nvec);
    const long expect_bucket2_size = pow(1.0 - alpha2 * alpha2, CSD * 0.5) * (main_bucket->num_pvec + main_bucket->num_nvec);
    for (long i = 0; i < batchsize; i++){
        dst3[i]->_alloc(0.5 * expect_bucket3_size, 1);
        dst3[i]->_alloc(0.5 * expect_bucket3_size, 0);
        if (for_bgj2) continue;
        dst2[i]->_alloc(0.5 * expect_bucket2_size, 1);
        dst2[i]->_alloc(0.5 * expect_bucket2_size, 0);
    }

    /////// choose centers ///////
    uint32_t center_ind_list[batchsize];
    uint8_t *center = (uint8_t *) NEW_VEC(batchsize * vec_length, sizeof(uint8_t));
    uint32_t center_ptr[batchsize];
    int32_t b2[batchsize];
    long num_try_find_center = 0;
    for (long i = 0; i < batchsize; i++) {
        int pass;
        do {
            pass = 1;
            num_try_find_center++;
            center_ind_list[i] = rand() % ((long)(0.65 * main_bucket->num_pvec));     // it really matters?
            for (long j = 0; j < i; j++) {
                if (center_ind_list[j] == center_ind_list[i] && num_try_find_center < 2000) pass = 0;
            }
            int8_t *ptr = vec + main_bucket->pvec[center_ind_list[i]] * vec_length;
            if (faraway_center) {
                if ((CSD * (int)ptr[0] * (int)ptr[0] > 2 * main_bucket->pnorm[center_ind_list[i]]) && (num_try_find_center < 1000)) pass = 0;
            }
            dst3[i]->set_center(main_bucket->pvec[center_ind_list[i]], main_bucket->pnorm[center_ind_list[i]], vu[main_bucket->pvec[center_ind_list[i]]]);
            if (!for_bgj2) dst2[i]->set_center(main_bucket->pvec[center_ind_list[i]], main_bucket->pnorm[center_ind_list[i]], vu[main_bucket->pvec[center_ind_list[i]]]);
            for (long j = 0; j < nb; j++) {
                __m256i x = _mm256_load_si256((__m256i *)(vec + main_bucket->pvec[center_ind_list[i]] * vec_length + j * 32));
                _mm256_store_si256((__m256i *)(center + i * vec_length + j * 32), _mm256_xor_si256(epi8_sign_bit, x));
            }
            b2[i] = goal_norm - dst3[i]->center_norm;
            center_ptr[i] = main_bucket->pvec[center_ind_list[i]];
        } while(!pass);
    }

    ///// prepare bucketing buffer /////
    #if SBUCKET_USE_BUFFER
    constexpr uint64_t csnp_ele_size = init_sieve ? 12 : 10;
    uint16_t *csnp3_pbuffer, *csnp3_nbuffer;                    // for bucket3, 12 or less Bytes for each vector
    uint16_t *csnp2_pbuffer, *csnp2_nbuffer;                    // for bucket2, 12 or less Bytes for each vector
    csnp3_pbuffer = (uint16_t *) malloc((SBUCKET_BUFFER_INIT_SIZE >> 1) * csnp_ele_size);
    csnp3_nbuffer = (uint16_t *) malloc((SBUCKET_BUFFER_INIT_SIZE >> 1) * csnp_ele_size);
    if (!for_bgj2) csnp2_pbuffer = (uint16_t *) malloc((SBUCKET_BUFFER_INIT_SIZE >> 1) * csnp_ele_size); 
    if (!for_bgj2) csnp2_nbuffer = (uint16_t *) malloc((SBUCKET_BUFFER_INIT_SIZE >> 1) * csnp_ele_size);
    long _csnp3_buffer_psize = SBUCKET_BUFFER_INIT_SIZE >> 1;
    long _csnp3_buffer_nsize = SBUCKET_BUFFER_INIT_SIZE >> 1;
    long _csnp2_buffer_psize = SBUCKET_BUFFER_INIT_SIZE >> 1;
    long _csnp2_buffer_nsize = SBUCKET_BUFFER_INIT_SIZE >> 1;
    long num_pcsnp3 = 0, num_ncsnp3 = 0, num_pcsnp2 = 0, num_ncsnp2 = 0;
    #endif
    
    /** 
     * \param[in] __p
     * \param[in] __cind    uint32_t
     * \param[in] __sum     int32_t
     * \param[in] __norm    uint32_t
     * \param[in] __dp      int32_t
     * \param[in] __ind     long 
    */
    #define _ADD_TO_CSNP3_BUFFER(__p, __cind, __sum, __norm, __ind)                         \
                                                                    do {                    \
        if (__p) {                                                                          \
            csnp3_pbuffer[num_pcsnp3*(csnp_ele_size>>1)] = __cind;                          \
            *((int16_t *)(&csnp3_pbuffer[num_pcsnp3*(csnp_ele_size>>1)+1])) = (__sum) >> 7; \
            if (init_sieve) {                                                               \
                *((int32_t *)(&csnp3_pbuffer[num_pcsnp3*(csnp_ele_size>>1)+2])) = __norm;   \
                *((uint32_t *)(&csnp3_pbuffer[num_pcsnp3*(csnp_ele_size>>1)+4])) = __ind;   \
            } else {                                                                        \
                csnp3_pbuffer[num_pcsnp3*(csnp_ele_size>>1)+2] = __norm;                    \
                *((uint32_t *)(&csnp3_pbuffer[num_pcsnp3*(csnp_ele_size>>1)+3])) = __ind;   \
            }                                                                               \
            num_pcsnp3++;                                                                   \
        } else {                                                                            \
            csnp3_nbuffer[num_ncsnp3*(csnp_ele_size>>1)] = __cind;                          \
            *((int16_t *)(&csnp3_nbuffer[num_ncsnp3*(csnp_ele_size>>1)+1])) = (__sum) >> 7; \
            if (init_sieve) {                                                               \
                *((int32_t *)(&csnp3_nbuffer[num_ncsnp3*(csnp_ele_size>>1)+2])) = __norm;   \
                *((uint32_t *)(&csnp3_nbuffer[num_ncsnp3*(csnp_ele_size>>1)+4])) = __ind;   \
            } else {                                                                        \
                csnp3_nbuffer[num_ncsnp3*(csnp_ele_size>>1)+2] = __norm;                    \
                *((uint32_t *)(&csnp3_nbuffer[num_ncsnp3*(csnp_ele_size>>1)+3])) = __ind;   \
            }                                                                               \
            num_ncsnp3++;                                                                   \
        }                                                                                   \
    } while (0)
    #define _ADD_TO_CSNP2_BUFFER(__p, __cind, __sum, __norm, __ind)                         \
                                                                    do {                    \
        if (__p) {                                                                          \
            csnp2_pbuffer[num_pcsnp2*(csnp_ele_size>>1)] = __cind;                          \
            *((int16_t *)(&csnp2_pbuffer[num_pcsnp2*(csnp_ele_size>>1)+1])) = (__sum) >> 7; \
            if (init_sieve) {                                                               \
                *((int32_t *)(&csnp2_pbuffer[num_pcsnp2*(csnp_ele_size>>1)+2])) = __norm;   \
                *((uint32_t *)(&csnp2_pbuffer[num_pcsnp2*(csnp_ele_size>>1)+4])) = __ind;   \
            } else {                                                                        \
                csnp2_pbuffer[num_pcsnp2*(csnp_ele_size>>1)+2] = __norm;                    \
                *((uint32_t *)(&csnp2_pbuffer[num_pcsnp2*(csnp_ele_size>>1)+3])) = __ind;   \
            }                                                                               \
            num_pcsnp2++;                                                                   \
        } else {                                                                            \
            csnp2_nbuffer[num_ncsnp2*(csnp_ele_size>>1)] = __cind;                          \
            *((int16_t *)(&csnp2_nbuffer[num_ncsnp2*(csnp_ele_size>>1)+1])) = (__sum) >> 7; \
            if (init_sieve) {                                                               \
                *((int32_t *)(&csnp2_nbuffer[num_ncsnp2*(csnp_ele_size>>1)+2])) = __norm;   \
                *((uint32_t *)(&csnp2_nbuffer[num_ncsnp2*(csnp_ele_size>>1)+4])) = __ind;   \
            } else {                                                                        \
                csnp2_nbuffer[num_ncsnp2*(csnp_ele_size>>1)+2] = __norm;                    \
                *((uint32_t *)(&csnp2_nbuffer[num_ncsnp2*(csnp_ele_size>>1)+3])) = __ind;   \
            }                                                                               \
            num_ncsnp2++;                                                                   \
        }                                                                                   \
    } while (0)
    #define ADD_TO_LOCAL_BUFFER(__p, __cind, __sum, __norm, __dp, __ind, __3bound)          \
                                                                                do {          \
        if (for_bgj2) {                                                                     \
            if (__p) {                                                                      \
                _ADD_TO_CSNP3_BUFFER(__p, __cind, __sum, __norm, main_bucket->pvec[__ind]); \
            } else {                                                                        \
                _ADD_TO_CSNP3_BUFFER(__p, __cind, __sum, __norm, main_bucket->nvec[__ind]); \
            }                                                                               \
            if (__p) {                                                                      \
                if ((int32_t) (__norm) - __dp < b2[__cind]) {                               \
                    TRY_ADDS_TO_DST(center_ptr[__cind], main_bucket->pvec[__ind]);          \
                }                                                                           \
            } else {                                                                        \
                if ((int32_t) (__norm) + __dp < b2[__cind]) {                               \
                    TRY_ADDA_TO_DST(center_ptr[__cind], main_bucket->nvec[__ind]);          \
                }                                                                           \
            }                                                                               \
        } else {                                                                            \
            if (__p) {                                                                      \
                _ADD_TO_CSNP2_BUFFER(__p, __cind, __sum, __norm, main_bucket->pvec[__ind]); \
            } else {                                                                        \
                _ADD_TO_CSNP2_BUFFER(__p, __cind, __sum, __norm, main_bucket->nvec[__ind]); \
            }                                                                               \
            if (((__dp > __3bound) && __p) || ((__3bound > __dp) && !(__p))) {              \
                if (__p) {                                                                  \
                    if ((int32_t) (__norm) - __dp < b2[__cind]) {                           \
                        TRY_ADDS_TO_DST(center_ptr[__cind], main_bucket->pvec[__ind]);      \
                    }                                                                       \
                } else {                                                                    \
                    if ((int32_t) (__norm) + __dp < b2[__cind]) {                           \
                        TRY_ADDA_TO_DST(center_ptr[__cind], main_bucket->nvec[__ind]);      \
                    }                                                                       \
                }                                                                           \
                if (__p) {                                                                      \
                    _ADD_TO_CSNP3_BUFFER(__p, __cind, __sum, __norm, main_bucket->pvec[__ind]); \
                } else {                                                                        \
                    _ADD_TO_CSNP3_BUFFER(__p, __cind, __sum, __norm, main_bucket->nvec[__ind]); \
                }                                                                               \
            }                                                                               \
        }                                                                                   \
    } while (0)
    #define PROCESS_CSNP3_BUFFER                                                                \
                                do {                                                            \
        for (long i = 0; i < num_pcsnp3; i++) {                                                 \
            if (init_sieve) {                                                                   \
                dst3[csnp3_pbuffer[i*(csnp_ele_size>>1)]]->add_pvec(                            \
                    *((uint32_t *)(&csnp3_pbuffer[i*(csnp_ele_size>>1)+4])),                    \
                    *((uint32_t *)(&csnp3_pbuffer[i*(csnp_ele_size>>1)+2])),                    \
                    ((int32_t) (*((int16_t *)(&csnp3_pbuffer[i*(csnp_ele_size>>1)+1])))) << 7   \
                );                                                                              \
            } else {                                                                            \
                dst3[csnp3_pbuffer[i*(csnp_ele_size>>1)]]->add_pvec(                            \
                    *((uint32_t *)(&csnp3_pbuffer[i*(csnp_ele_size>>1)+3])),                    \
                    csnp3_pbuffer[i*(csnp_ele_size>>1)+2],                                      \
                    ((int32_t) (*((int16_t *)(&csnp3_pbuffer[i*(csnp_ele_size>>1)+1])))) << 7   \
                );                                                                              \
            }                                                                                   \
        }                                                                                       \
        for (long i = 0; i < num_ncsnp3; i++) {                                                 \
            if (init_sieve) {                                                                   \
                dst3[csnp3_nbuffer[i*(csnp_ele_size>>1)]]->add_nvec(                            \
                    *((uint32_t *)(&csnp3_nbuffer[i*(csnp_ele_size>>1)+4])),                    \
                    *((uint32_t *)(&csnp3_nbuffer[i*(csnp_ele_size>>1)+2])),                    \
                    ((int32_t) (*((int16_t *)(&csnp3_nbuffer[i*(csnp_ele_size>>1)+1])))) << 7   \
                );                                                                              \
            } else {                                                                            \
                dst3[csnp3_nbuffer[i*(csnp_ele_size>>1)]]->add_nvec(                            \
                    *((uint32_t *)(&csnp3_nbuffer[i*(csnp_ele_size>>1)+3])),                    \
                    csnp3_nbuffer[i*(csnp_ele_size>>1)+2],                                      \
                    ((int32_t) (*((int16_t *)(&csnp3_nbuffer[i*(csnp_ele_size>>1)+1])))) << 7   \
                );                                                                              \
            }                                                                                   \
        }                                                                                       \
        num_pcsnp3 = 0;                                                                         \
        num_ncsnp3 = 0;                                                                         \
    } while (0)
    #define PROCESS_CSNP2_BUFFER                                                                \
                                do {                                                            \
        for (long i = 0; i < num_pcsnp2; i++) {                                                 \
            if (init_sieve) {                                                                   \
                dst2[csnp2_pbuffer[i*(csnp_ele_size>>1)]]->add_pvec(                            \
                    *((uint32_t *)(&csnp2_pbuffer[i*(csnp_ele_size>>1)+4])),                    \
                    *((uint32_t *)(&csnp2_pbuffer[i*(csnp_ele_size>>1)+2])),                    \
                    ((int32_t) (*((int16_t *)(&csnp2_pbuffer[i*(csnp_ele_size>>1)+1])))) << 7   \
                );                                                                              \
            } else {                                                                            \
                dst2[csnp2_pbuffer[i*(csnp_ele_size>>1)]]->add_pvec(                            \
                    *((uint32_t *)(&csnp2_pbuffer[i*(csnp_ele_size>>1)+3])),                    \
                    csnp2_pbuffer[i*(csnp_ele_size>>1)+2],                                      \
                    ((int32_t) (*((int16_t *)(&csnp2_pbuffer[i*(csnp_ele_size>>1)+1])))) << 7   \
                );                                                                              \
            }                                                                                   \
        }                                                                                       \
        for (long i = 0; i < num_ncsnp2; i++) {                                                 \
            if (init_sieve) {                                                                   \
                dst2[csnp2_nbuffer[i*(csnp_ele_size>>1)]]->add_nvec(                            \
                    *((uint32_t *)(&csnp2_nbuffer[i*(csnp_ele_size>>1)+4])),                    \
                    *((uint32_t *)(&csnp2_nbuffer[i*(csnp_ele_size>>1)+2])),                    \
                    ((int32_t) (*((int16_t *)(&csnp2_nbuffer[i*(csnp_ele_size>>1)+1])))) << 7   \
                );                                                                              \
            } else {                                                                            \
                dst2[csnp2_nbuffer[i*(csnp_ele_size>>1)]]->add_nvec(                            \
                    *((uint32_t *)(&csnp2_nbuffer[i*(csnp_ele_size>>1)+3])),                    \
                    csnp2_nbuffer[i*(csnp_ele_size>>1)+2],                                      \
                    ((int32_t) (*((int16_t *)(&csnp2_nbuffer[i*(csnp_ele_size>>1)+1])))) << 7   \
                );                                                                              \
            }                                                                                   \
        }                                                                                       \
        num_pcsnp2 = 0;                                                                         \
        num_ncsnp2 = 0;                                                                         \
    } while (0)
    #define PROCESS_AND_FREE_LOCAL_BUFFER   \
                                    do {    \
        PROCESS_CSNP3_BUFFER;               \
        if (!for_bgj2) PROCESS_CSNP2_BUFFER;\
        free(csnp3_pbuffer);                \
        free(csnp3_nbuffer);                \
        if (!for_bgj2) free(csnp2_pbuffer); \
        if (!for_bgj2) free(csnp2_nbuffer); \
    } while(0)
    #define CHECK_BUFFER_FULL                                                                                       \
                                do {                                                                                  \
        if ((num_pcsnp3 + 8 * batchsize > _csnp3_buffer_psize) ||                                                   \
            (num_ncsnp3 + 8 * batchsize > _csnp3_buffer_nsize)) {                                                   \
            if (SBUCKET_BUFFER_SIZE_DYNAMIC) {                                                                      \
                csnp3_pbuffer = (uint16_t *) realloc(csnp3_pbuffer, _csnp3_buffer_psize * 2 * csnp_ele_size);       \
                csnp3_nbuffer = (uint16_t *) realloc(csnp3_nbuffer, _csnp3_buffer_nsize * 2 * csnp_ele_size);       \
                _csnp3_buffer_psize *= 2;                                                                           \
                _csnp3_buffer_nsize *= 2;                                                                           \
            } else {                                                                                                \
                PROCESS_CSNP3_BUFFER;                                                                               \
            }                                                                                                       \
        }                                                                                                           \
                                                                                                                    \
        if (!for_bgj2) {                                                                                            \
            if ((num_pcsnp2 + 8 * batchsize > _csnp2_buffer_psize) ||                                               \
                (num_ncsnp2 + 8 * batchsize > _csnp2_buffer_nsize)) {                                               \
                if (SBUCKET_BUFFER_SIZE_DYNAMIC) {                                                                  \
                    csnp2_pbuffer = (uint16_t *) realloc(csnp2_pbuffer, _csnp2_buffer_psize * 2 * csnp_ele_size);   \
                    csnp2_nbuffer = (uint16_t *) realloc(csnp2_nbuffer, _csnp2_buffer_nsize * 2 * csnp_ele_size);   \
                    _csnp2_buffer_psize *= 2;                                                                       \
                    _csnp2_buffer_nsize *= 2;                                                                       \
                } else {                                                                                            \
                    PROCESS_CSNP2_BUFFER;                                                                           \
                }                                                                                                   \
            }                                                                                                       \
        }                                                                                                           \
    } while(0)

    #define ADD_TO_BUCKET(__p, __cind, __sum, __norm, __dp, __ind, __3bound)                \
                                                                                do {        \
        if (for_bgj2) {                                                                     \
            if (__p) {                                                                      \
                dst3[__cind]->add_pvec(main_bucket->pvec[__ind], __norm, __sum);            \
            } else {                                                                        \
                dst3[__cind]->add_nvec(main_bucket->nvec[__ind], __norm, __sum);            \
            }                                                                               \
            if (__p) {                                                                      \
                if ((int32_t) (__norm) - __dp < b2[__cind]) {                               \
                    TRY_ADDS_TO_DST(center_ptr[__cind], main_bucket->pvec[__ind]);          \
                }                                                                           \
            } else {                                                                        \
                if ((int32_t) (__norm) + __dp < b2[__cind]) {                               \
                    TRY_ADDA_TO_DST(center_ptr[__cind], main_bucket->nvec[__ind]);          \
                }                                                                           \
            }                                                                               \
        } else {                                                                            \
            if (__p) {                                                                      \
                dst2[__cind]->add_pvec(main_bucket->pvec[__ind], __norm, __sum);            \
            } else {                                                                        \
                dst2[__cind]->add_nvec(main_bucket->nvec[__ind], __norm, __sum);            \
            }                                                                               \
            if (((__dp > __3bound) && __p) || ((__3bound > __dp) && !(__p))) {              \
                if (__p) {                                                                  \
                    if ((int32_t) (__norm) - __dp < b2[__cind]) {                           \
                        TRY_ADDS_TO_DST(center_ptr[__cind], main_bucket->pvec[__ind]);      \
                    }                                                                       \
                } else {                                                                    \
                    if ((int32_t) (__norm) + __dp < b2[__cind]) {                           \
                        TRY_ADDA_TO_DST(center_ptr[__cind], main_bucket->nvec[__ind]);      \
                    }                                                                       \
                }                                                                           \
                if (__p) {                                                                  \
                    dst3[__cind]->add_pvec(main_bucket->pvec[__ind], __norm, __sum);        \
                } else {                                                                    \
                    dst3[__cind]->add_nvec(main_bucket->nvec[__ind], __norm, __sum);        \
                }                                                                           \
            }                                                                               \
        }                                                                                   \
    } while (0)
    
    ///// main loop //////
    long pnd = 0;
    while (pnd < main_bucket->num_pvec - 7) {
        __m256i sum_b8 = _mm256_loadu_si256((__m256i *)(main_bucket->psum + pnd));
        __m256i norm_b8 = _mm256_loadu_si256((__m256i *)(main_bucket->pnorm + pnd));
        __m256i bound3, bound2;
        int32_t *sum_epi32 = (int32_t *)(&sum_b8);
        uint32_t *norm_epi32 = (uint32_t *)(&norm_b8);
        int32_t *bound3_epi32 = (int32_t *)(&bound3);
        int32_t *bound2_epi32 = (int32_t *)(&bound2);
        if (init_sieve) {
            __m256i norm_hi = _mm256_srai_epi64(norm_b8, 32);
            __m256i bound3_lo = _mm256_srai_epi64(_mm256_mul_epi32(norm_b8, alpha3x2_si256), 16);
            __m256i bound3_hi = _mm256_slli_epi64(_mm256_mul_epi32(norm_hi, alpha3x2_si256), 16);
            if (!for_bgj2) {
                __m256i bound2_lo = _mm256_srai_epi64(_mm256_mul_epi32(norm_b8, alpha2x2_si256), 16);
                __m256i bound2_hi = _mm256_slli_epi64(_mm256_mul_epi32(norm_hi, alpha2x2_si256), 16);
                bound2 = _mm256_blend_epi32(bound2_lo, bound2_hi, 170);
            }
            bound3 = _mm256_blend_epi32(bound3_lo, bound3_hi, 170);
        } else {
            bound3 = _mm256_mulhi_epu16(norm_b8, alpha3x2_si256);
            if (!for_bgj2) bound2 = _mm256_mulhi_epu16(norm_b8, alpha2x2_si256);
        }

        __m256i dst[batchsize];             // record the dot products
        int32_t *dst_epi32 = (int32_t *) (&dst[0]);
        uint64_t cdst[batchsize >> 3];      // record the cmp results
        uint32_t cmp[8];

        __attribute__ ((aligned (32))) int8_t s[vec_length * 8];
        for (long j = 0; j < vec_length; j += 32) {
            _mm256_store_si256((__m256i *)(s+0*vec_length+j), _mm256_load_si256((__m256i *)(vec + main_bucket->pvec[pnd+0] * vec_length + j)));
            _mm256_store_si256((__m256i *)(s+1*vec_length+j), _mm256_load_si256((__m256i *)(vec + main_bucket->pvec[pnd+1] * vec_length + j)));
            _mm256_store_si256((__m256i *)(s+2*vec_length+j), _mm256_load_si256((__m256i *)(vec + main_bucket->pvec[pnd+2] * vec_length + j)));
            _mm256_store_si256((__m256i *)(s+3*vec_length+j), _mm256_load_si256((__m256i *)(vec + main_bucket->pvec[pnd+3] * vec_length + j)));
            _mm256_store_si256((__m256i *)(s+4*vec_length+j), _mm256_load_si256((__m256i *)(vec + main_bucket->pvec[pnd+4] * vec_length + j)));
            _mm256_store_si256((__m256i *)(s+5*vec_length+j), _mm256_load_si256((__m256i *)(vec + main_bucket->pvec[pnd+5] * vec_length + j)));
            _mm256_store_si256((__m256i *)(s+6*vec_length+j), _mm256_load_si256((__m256i *)(vec + main_bucket->pvec[pnd+6] * vec_length + j)));
            _mm256_store_si256((__m256i *)(s+7*vec_length+j), _mm256_load_si256((__m256i *)(vec + main_bucket->pvec[pnd+7] * vec_length + j)));
        }
        
        if (pnd < main_bucket->num_pvec - 23) {
            for (long i = 8; i < 16; i++) {
                for (long j = 0; j < vec_length; j += 64) {
                    _mm_prefetch(vec + main_bucket->pvec[pnd+i] * vec_length + j, _MM_HINT_T0);
                }
            }
            _mm_prefetch(main_bucket->psum + pnd + 8, _MM_HINT_T0);
            _mm_prefetch(main_bucket->pnorm + pnd + 8, _MM_HINT_T0);
            _mm_prefetch(main_bucket->pvec + pnd + 16, _MM_HINT_T0);
        }

        uint32_t cind = 0;
        while (cind < batchsize) {
            vdp8x8(&dst[cind], center + cind * vec_length, s);
            for (long i = 0; i < 8; i++) dst[cind+i] = _mm256_sub_epi32(dst[cind+i], sum_b8);
            if (for_bgj2) {
                for (long i = 0; i < 8; i++) cmp[i] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(dst[cind+i], bound3));
            } else {
                for (long i = 0; i < 8; i++) cmp[i] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(dst[cind+i], bound2));
            }
            uint64_t cdst_lo = ( cmp[0] | (cmp[1] << 8) | (cmp[2] << 16) | (cmp[3] << 24) );
            uint64_t cdst_hi = ( cmp[4] | (cmp[5] << 8) | (cmp[6] << 16) | (cmp[7] << 24) );
            cdst[cind >> 3] = ( cdst_lo | (cdst_hi << 32) );
            cind += 8;
        }

        for (uint32_t i = 0; i < batchsize; i += 8) {
            while(cdst[i>>3]) {
                uint32_t r = __builtin_ctzl(cdst[i>>3]);
                cdst[i>>3] -= (1UL << r);
                #if SBUCKET_USE_BUFFER
                ADD_TO_LOCAL_BUFFER(1, i+(r>>3), sum_epi32[r&(0x7)], norm_epi32[r&(0x7)], dst_epi32[i*8+r], pnd+(r&(0x7)), bound3_epi32[r&(0x7)]);
                #else
                ADD_TO_BUCKET(1, i+(r>>3), sum_epi32[r&(0x7)], norm_epi32[r&(0x7)], dst_epi32[i*8+r], pnd+(r&(0x7)), bound3_epi32[r&(0x7)]);
                #endif
            }
        }

        #if SBUCKET_USE_BUFFER
        CHECK_BUFFER_FULL;
        #endif

        pnd += 8;
    }
    if (pnd < main_bucket->num_pvec) {
        const long prem = main_bucket->num_pvec - pnd;

        __m256i sum_si256[7];
        __m256i bound2_si256[7];
        __m256i bound3_si256[7];
        int32_t *sum_epi32 = main_bucket->psum + pnd;
        int32_t *norm_epi32 = main_bucket->pnorm + pnd;
        int32_t *bound3_epi32 = (int32_t *)&bound3_si256[0];
        for (long i = 0; i < prem; i++) sum_si256[i] = _mm256_set1_epi32(sum_epi32[i]);
        for (long i = 0; i < prem; i++) {
            bound3_si256[i] = _mm256_set1_epi32(((int64_t) norm_epi32[i] * (int64_t) alpha3x2_epu16) >> 16);
            if (!for_bgj2) bound2_si256[i] = _mm256_set1_epi32(((int64_t) norm_epi32[i] * (int64_t) alpha2x2_epu16) >> 16);
        }

        __m256i dst[7 * (batchsize>>3)];
        int32_t *dst_epi32 = (int32_t *) (&dst[0]);
        uint64_t cdst[batchsize >> 3];
        uint32_t cmp[8] = {};
        
        uint32_t cind = 0;
        while (cind < batchsize) {
            long bias = prem * (cind >> 3);
            vdp8xn(&dst[bias], main_bucket->pvec + pnd, center + cind * vec_length, prem);
            for (long i = 0; i < prem; i++) dst[bias+i] = _mm256_sub_epi32(dst[bias+i], sum_si256[i]);
            if (for_bgj2) {
                for (long i = 0; i < prem; i++) cmp[i] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(dst[bias+i], bound3_si256[i]));
            } else {
                for (long i = 0; i < prem; i++) cmp[i] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(dst[bias+i], bound2_si256[i]));
            }
            uint64_t cdst_lo = ( cmp[0] | (cmp[1] << 8) | (cmp[2] << 16) | (cmp[3] << 24) );
            if (prem > 3) {
                uint64_t cdst_hi = ( cmp[4] | (cmp[5] << 8) | (cmp[6] << 16) | (cmp[7] << 24) );
                cdst[cind >> 3] = ( cdst_lo | (cdst_hi << 32) );
            } else {
                cdst[cind >> 3] = cdst_lo;
            }
            
            cind += 8;
        }

        for (uint32_t i = 0; i < batchsize; i += 8) {
            while(cdst[i>>3]) {
                uint32_t r = __builtin_ctzl(cdst[i>>3]);
                cdst[i>>3] -= (1UL << r);
                #if SBUCKET_USE_BUFFER
                ADD_TO_LOCAL_BUFFER(1, i+(r & 0x7), sum_epi32[r>>3], norm_epi32[r >> 3], dst_epi32[i*prem+r], pnd+(r>>3), bound3_epi32[r]);
                #else
                ADD_TO_BUCKET(1, i+(r & 0x7), sum_epi32[r>>3], norm_epi32[r >> 3], dst_epi32[i*prem+r], pnd+(r>>3), bound3_epi32[r]);
                #endif
            }
        }
    }

    long nnd = 0;
    while (nnd < main_bucket->num_nvec - 7) {
        __m256i sum_b8 = _mm256_loadu_si256((__m256i *)(main_bucket->nsum + nnd));
        __m256i norm_b8 = _mm256_loadu_si256((__m256i *)(main_bucket->nnorm + nnd));
        __m256i bound3, bound2;
        int32_t *sum_epi32 = (int32_t *)(&sum_b8);
        uint32_t *norm_epi32 = (uint32_t *)(&norm_b8);
        int32_t *bound3_epi32 = (int32_t *)(&bound3);
        int32_t *bound2_epi32 = (int32_t *)(&bound2);
        if (init_sieve) {
            __m256i norm_hi = _mm256_srai_epi64(norm_b8, 32);
            __m256i bound3_lo = _mm256_srai_epi64(_mm256_mul_epi32(norm_b8, alpha3x2_si256), 16);
            __m256i bound3_hi = _mm256_slli_epi64(_mm256_mul_epi32(norm_hi, alpha3x2_si256), 16);
            if (!for_bgj2) {
                __m256i bound2_lo = _mm256_srai_epi64(_mm256_mul_epi32(norm_b8, alpha2x2_si256), 16);
                __m256i bound2_hi = _mm256_slli_epi64(_mm256_mul_epi32(norm_hi, alpha2x2_si256), 16);
                bound2 = _mm256_blend_epi32(bound2_lo, bound2_hi, 170);
            }
            bound3 = _mm256_blend_epi32(bound3_lo, bound3_hi, 170);
        } else {
            bound3 = _mm256_mulhi_epu16(norm_b8, alpha3x2_si256);
            if (!for_bgj2) bound2 = _mm256_mulhi_epu16(norm_b8, alpha2x2_si256);
        }
        bound3 = _mm256_sub_epi32(_mm256_setzero_si256(), bound3);
        if (!for_bgj2) bound2 = _mm256_sub_epi32(_mm256_setzero_si256(), bound2);

        __m256i dst[batchsize];             // record the dot products
        int32_t *dst_epi32 = (int32_t *) (&dst[0]);
        uint64_t cdst[batchsize >> 3];      // record the cmp results
        uint32_t cmp[8];

        __attribute__ ((aligned (32))) int8_t s[vec_length * 8];
        for (long j = 0; j < vec_length; j += 32) {
            _mm256_store_si256((__m256i *)(s+0*vec_length+j), _mm256_load_si256((__m256i *)(vec + main_bucket->nvec[nnd+0] * vec_length + j)));
            _mm256_store_si256((__m256i *)(s+1*vec_length+j), _mm256_load_si256((__m256i *)(vec + main_bucket->nvec[nnd+1] * vec_length + j)));
            _mm256_store_si256((__m256i *)(s+2*vec_length+j), _mm256_load_si256((__m256i *)(vec + main_bucket->nvec[nnd+2] * vec_length + j)));
            _mm256_store_si256((__m256i *)(s+3*vec_length+j), _mm256_load_si256((__m256i *)(vec + main_bucket->nvec[nnd+3] * vec_length + j)));
            _mm256_store_si256((__m256i *)(s+4*vec_length+j), _mm256_load_si256((__m256i *)(vec + main_bucket->nvec[nnd+4] * vec_length + j)));
            _mm256_store_si256((__m256i *)(s+5*vec_length+j), _mm256_load_si256((__m256i *)(vec + main_bucket->nvec[nnd+5] * vec_length + j)));
            _mm256_store_si256((__m256i *)(s+6*vec_length+j), _mm256_load_si256((__m256i *)(vec + main_bucket->nvec[nnd+6] * vec_length + j)));
            _mm256_store_si256((__m256i *)(s+7*vec_length+j), _mm256_load_si256((__m256i *)(vec + main_bucket->nvec[nnd+7] * vec_length + j)));
        }

        if (nnd < main_bucket->num_nvec - 23) {
            for (long i = 8; i < 16; i++) {
                for (long j = 0; j < vec_length; j += 64) {
                    _mm_prefetch(vec + main_bucket->nvec[nnd+i] * vec_length + j, _MM_HINT_T0);
                }
            }
            _mm_prefetch(main_bucket->nsum + nnd + 8, _MM_HINT_T0);
            _mm_prefetch(main_bucket->nnorm + nnd + 8, _MM_HINT_T0);
            _mm_prefetch(main_bucket->nvec + nnd + 16, _MM_HINT_T0);
        }

        uint32_t cind = 0;
        while (cind < batchsize) {
            vdp8x8(&dst[cind], center + cind * vec_length, s);
            for (long i = 0; i < 8; i++) dst[cind+i] = _mm256_sub_epi32(dst[cind+i], sum_b8);
            if (for_bgj2) {
                for (long i = 0; i < 8; i++) cmp[i] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(bound3, dst[cind+i]));
            } else {
                for (long i = 0; i < 8; i++) cmp[i] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(bound2, dst[cind+i]));
            }
            uint64_t cdst_lo = ( cmp[0] | (cmp[1] << 8) | (cmp[2] << 16) | (cmp[3] << 24) );
            uint64_t cdst_hi = ( cmp[4] | (cmp[5] << 8) | (cmp[6] << 16) | (cmp[7] << 24) );
            cdst[cind >> 3] = ( cdst_lo | (cdst_hi << 32) );
            cind += 8;
        }

        for (uint32_t i = 0; i < batchsize; i += 8) {
            while(cdst[i>>3]) {
                uint32_t r = __builtin_ctzl(cdst[i>>3]);
                cdst[i>>3] -= (1UL << r);
                #if SBUCKET_USE_BUFFER
                ADD_TO_LOCAL_BUFFER(0, i+(r>>3), sum_epi32[r&(0x7)], norm_epi32[r&(0x7)], dst_epi32[i*8+r], nnd+(r&(0x7)), bound3_epi32[r&(0x7)]);
                #else
                ADD_TO_BUCKET(0, i+(r>>3), sum_epi32[r&(0x7)], norm_epi32[r&(0x7)], dst_epi32[i*8+r], nnd+(r&(0x7)), bound3_epi32[r&(0x7)]);
                #endif
            }
        }

        #if SBUCKET_USE_BUFFER
        CHECK_BUFFER_FULL;
        #endif

        nnd += 8;
    }
    if (nnd < main_bucket->num_nvec) {
        const long nrem = main_bucket->num_nvec - nnd;

        __m256i sum_si256[7];
        __m256i bound2_si256[7];
        __m256i bound3_si256[7];
        int32_t *sum_epi32 = main_bucket->nsum + nnd;
        int32_t *norm_epi32 = main_bucket->nnorm + nnd;
        int32_t *bound3_epi32 = (int32_t *)&bound3_si256[0];
        for (long i = 0; i < nrem; i++) sum_si256[i] = _mm256_set1_epi32(sum_epi32[i]);
        for (long i = 0; i < nrem; i++) {
            bound3_si256[i] = _mm256_set1_epi32(-(((int64_t) norm_epi32[i] * (int64_t) alpha3x2_epu16) >> 16));
            if (!for_bgj2) bound2_si256[i] = _mm256_set1_epi32(-(((int64_t) norm_epi32[i] * (int64_t) alpha2x2_epu16) >> 16));
        }

        __m256i dst[7 * (batchsize>>3)];
        int32_t *dst_epi32 = (int32_t *) (&dst[0]);
        uint64_t cdst[batchsize >> 3];
        uint32_t cmp[8] = {};
        
        uint32_t cind = 0;
        while (cind < batchsize) {
            long bias = nrem * (cind >> 3);
            vdp8xn(&dst[bias], main_bucket->nvec + nnd, center + cind * vec_length, nrem);
            for (long i = 0; i < nrem; i++) dst[bias+i] = _mm256_sub_epi32(dst[bias+i], sum_si256[i]);
            if (for_bgj2) {
                for (long i = 0; i < nrem; i++) cmp[i] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(bound3_si256[i], dst[bias+i]));
            } else {
                for (long i = 0; i < nrem; i++) cmp[i] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(bound2_si256[i], dst[bias+i]));
            }
            uint64_t cdst_lo = ( cmp[0] | (cmp[1] << 8) | (cmp[2] << 16) | (cmp[3] << 24) );
            if (nrem > 3) {
                uint64_t cdst_hi = ( cmp[4] | (cmp[5] << 8) | (cmp[6] << 16) | (cmp[7] << 24) );
                cdst[cind >> 3] = ( cdst_lo | (cdst_hi << 32) );
            } else {
                cdst[cind >> 3] = cdst_lo;
            }
            
            cind += 8;
        }

        for (uint32_t i = 0; i < batchsize; i += 8) {
            while(cdst[i>>3]) {
                uint32_t r = __builtin_ctzl(cdst[i>>3]);
                cdst[i>>3] -= (1UL << r);
                #if SBUCKET_USE_BUFFER
                ADD_TO_LOCAL_BUFFER(0, i+(r & 0x7), sum_epi32[r>>3], norm_epi32[r >> 3], dst_epi32[i*nrem+r], nnd+(r>>3), bound3_epi32[r]);
                #else
                ADD_TO_BUCKET(0, i+(r & 0x7), sum_epi32[r>>3], norm_epi32[r >> 3], dst_epi32[i*nrem+r], nnd+(r>>3), bound3_epi32[r]);
                #endif
            }
        }
    }

    #if SBUCKET_USE_BUFFER
    PROCESS_AND_FREE_LOCAL_BUFFER;
    #endif

    /////// remove center in the buckets ///////
    #pragma omp parallel for
    for (long i = 0; i < batchsize; i++) {
        if (for_bgj2) dst3[i]->remove_center(1);
        if (!for_bgj2) dst3[i]->remove_center(2);
        if (!for_bgj2) dst2[i]->remove_center(2);
    }

    /////// remove duplicates ///////
    for (long i = 0; i < batchsize; i++) {
        for (long j = 0; j < i; j++) {
            if (center_ind_list[i] == center_ind_list[j]) {
                delete dst3[i];
                dst3[i] = NULL;
                if (!for_bgj2) {
                    delete dst2[i];
                    dst2[i] = NULL;
                }
                break;
            }
        }
    }

    /////// free center ///////
    FREE_VEC((void *) center);

    if (profiling && prof) {
        pthread_spin_lock(&prof->profile_lock);
        prof->try_add2 += try_add2;
        prof->try_add3 += try_add3;
        prof->succ_add2 += succ_add2;
        prof->succ_add3 += succ_add3;
        pthread_spin_unlock(&prof->profile_lock);
    } 

    return 1;
}

template <uint32_t nb>
template <bool record_dp, bool profiling>
int Pool_epi8_t<nb>::_search_cred(bucket_epi8_t<record_dp> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<nb> *prof) { 
    if (!record_dp) return 0;
    __m256i b2 = _mm256_set1_epi32(goal_norm - bkt->center_norm);

    uint64_t try_add2 = 0;
    uint64_t try_add3 = 0;
    uint64_t succ_add2 = 0;
    uint64_t succ_add3 = 0;
    
    long ind = 0;
    while (ind < bkt->num_pvec - 63) {
        __m256i *nptr = (__m256i *) (bkt->pnorm + ind);
        __m256i *dptr = (__m256i *) (bkt->pdot + ind);
        _mm256_storeu_si256(dptr+0, _mm256_sub_epi32(_mm256_loadu_si256(nptr+0), _mm256_loadu_si256(dptr+0)));
        _mm256_storeu_si256(dptr+1, _mm256_sub_epi32(_mm256_loadu_si256(nptr+1), _mm256_loadu_si256(dptr+1)));
        _mm256_storeu_si256(dptr+2, _mm256_sub_epi32(_mm256_loadu_si256(nptr+2), _mm256_loadu_si256(dptr+2)));
        _mm256_storeu_si256(dptr+3, _mm256_sub_epi32(_mm256_loadu_si256(nptr+3), _mm256_loadu_si256(dptr+3)));
        _mm256_storeu_si256(dptr+4, _mm256_sub_epi32(_mm256_loadu_si256(nptr+4), _mm256_loadu_si256(dptr+4)));
        _mm256_storeu_si256(dptr+5, _mm256_sub_epi32(_mm256_loadu_si256(nptr+5), _mm256_loadu_si256(dptr+5)));
        _mm256_storeu_si256(dptr+6, _mm256_sub_epi32(_mm256_loadu_si256(nptr+6), _mm256_loadu_si256(dptr+6)));
        _mm256_storeu_si256(dptr+7, _mm256_sub_epi32(_mm256_loadu_si256(nptr+7), _mm256_loadu_si256(dptr+7)));
        uint32_t cmp0 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+0)));
        uint32_t cmp1 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+1)));
        uint32_t cmp2 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+2)));
        uint32_t cmp3 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+3)));
        uint32_t cmp4 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+4)));
        uint32_t cmp5 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+5)));
        uint32_t cmp6 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+6)));
        uint32_t cmp7 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+7)));
        
        uint64_t cdst_lo = ( cmp0 | (cmp1 << 8) | (cmp2 << 16) | (cmp3 << 24) );
        uint64_t cdst_hi = ( cmp4 | (cmp5 << 8) | (cmp6 << 16) | (cmp7 << 24) );
        uint64_t cdst = ( cdst_lo | (cdst_hi << 32) );
        CHECK_AND_ADD_CRED_1X64(cdst, (bkt->pvec + ind), TRY_ADDS_TO_DST);
        ind += 64;
    }
    
    if (ind < bkt->num_pvec) {
        uint64_t cdst = 0;
        long i = 0;
        while (ind + i < bkt->num_pvec - 7) {
            __m256i *nptr = (__m256i *) (bkt->pnorm + ind + i);
            __m256i *dptr = (__m256i *) (bkt->pdot + ind + i);
            _mm256_storeu_si256(dptr, _mm256_sub_epi32(_mm256_loadu_si256(nptr), _mm256_loadu_si256(dptr)));
            uint64_t cmp = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+0)));
            cdst |= (cmp << i);
            i += 8;
        }
        while (ind + i < bkt->num_pvec) {
            bkt->pdot[ind+i] = bkt->pnorm[ind+i] - bkt->pdot[ind+i];
            if (goal_norm - bkt->center_norm > bkt->pdot[ind+i]) {
                cdst |= (1UL << i);
            }
            i++;
        }
        CHECK_AND_ADD_CRED_1X64(cdst, (bkt->pvec + ind), TRY_ADDS_TO_DST);
    }

    ind = 0;
    while (ind < bkt->num_nvec - 63) {
        __m256i *nptr = (__m256i *) (bkt->nnorm + ind);
        __m256i *dptr = (__m256i *) (bkt->ndot + ind);
        _mm256_storeu_si256(dptr+0, _mm256_add_epi32(_mm256_loadu_si256(nptr+0), _mm256_loadu_si256(dptr+0)));
        _mm256_storeu_si256(dptr+1, _mm256_add_epi32(_mm256_loadu_si256(nptr+1), _mm256_loadu_si256(dptr+1)));
        _mm256_storeu_si256(dptr+2, _mm256_add_epi32(_mm256_loadu_si256(nptr+2), _mm256_loadu_si256(dptr+2)));
        _mm256_storeu_si256(dptr+3, _mm256_add_epi32(_mm256_loadu_si256(nptr+3), _mm256_loadu_si256(dptr+3)));
        _mm256_storeu_si256(dptr+4, _mm256_add_epi32(_mm256_loadu_si256(nptr+4), _mm256_loadu_si256(dptr+4)));
        _mm256_storeu_si256(dptr+5, _mm256_add_epi32(_mm256_loadu_si256(nptr+5), _mm256_loadu_si256(dptr+5)));
        _mm256_storeu_si256(dptr+6, _mm256_add_epi32(_mm256_loadu_si256(nptr+6), _mm256_loadu_si256(dptr+6)));
        _mm256_storeu_si256(dptr+7, _mm256_add_epi32(_mm256_loadu_si256(nptr+7), _mm256_loadu_si256(dptr+7)));
        uint32_t cmp0 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+0)));
        uint32_t cmp1 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+1)));
        uint32_t cmp2 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+2)));
        uint32_t cmp3 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+3)));
        uint32_t cmp4 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+4)));
        uint32_t cmp5 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+5)));
        uint32_t cmp6 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+6)));
        uint32_t cmp7 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+7)));

        uint64_t cdst_lo = ( cmp0 | (cmp1 << 8) | (cmp2 << 16) | (cmp3 << 24) );
        uint64_t cdst_hi = ( cmp4 | (cmp5 << 8) | (cmp6 << 16) | (cmp7 << 24) );
        uint64_t cdst = ( cdst_lo | (cdst_hi << 32) );
        CHECK_AND_ADD_CRED_1X64(cdst, (bkt->nvec + ind), TRY_ADDA_TO_DST);
        ind += 64;
    }
    
    if (ind < bkt->num_nvec) {
        uint64_t cdst = 0;
        long i = 0;
        while (ind + i < bkt->num_nvec - 7) {
            __m256i *nptr = (__m256i *) (bkt->nnorm + ind + i);
            __m256i *dptr = (__m256i *) (bkt->ndot + ind + i);
            _mm256_storeu_si256(dptr, _mm256_add_epi32(_mm256_loadu_si256(nptr), _mm256_loadu_si256(dptr)));
            uint64_t cmp = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_loadu_si256(dptr+0)));
            cdst |= (cmp << i);
            i += 8;
        }
        while (ind + i < bkt->num_nvec) {
            bkt->ndot[ind+i] = bkt->nnorm[ind+i] + bkt->ndot[ind+i];
            if (goal_norm - bkt->center_norm > bkt->ndot[ind+i]) {
                cdst |= (1UL << i);
            }
            i++;
        }
        CHECK_AND_ADD_CRED_1X64(cdst, (bkt->nvec + ind), TRY_ADDA_TO_DST);
    }

    if (profiling && prof) {
        pthread_spin_lock(&prof->profile_lock);
        prof->try_add2 += try_add2;
        prof->try_add3 += try_add3;
        prof->succ_add2 += succ_add2;
        prof->succ_add3 += succ_add3;
        pthread_spin_unlock(&prof->profile_lock);
    } 

    return 1;
}

template <uint32_t nb>
template <uint32_t l2_block, uint32_t l1_block, bool record_dp, bool profiling>
int Pool_epi8_t<nb>::_search_np(bucket_epi8_t<record_dp> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<nb> *prof) {
    uint64_t try_add2 = 0;
    uint64_t try_add3 = 0;
    uint64_t succ_add2 = 0;
    uint64_t succ_add3 = 0;

    const __m256i gn = _mm256_set1_epi32(goal_norm);
    const __m256i cn = _mm256_set1_epi32(goal_norm - bkt->center_norm);

    for (long Pnd = 0; Pnd < bkt->num_pvec; Pnd += l2_block) {
        for (long Nnd = 0; Nnd < bkt->num_nvec; Nnd += l2_block) {
            const long Pbound = (Pnd + l2_block > bkt->num_pvec) ? bkt->num_pvec : Pnd + l2_block;
            const long Nbound = (Nnd + l2_block > bkt->num_nvec) ? bkt->num_nvec : Nnd + l2_block;

            long pnd = Pnd;
            while (pnd < Pbound - l1_block + 1) {
                constexpr long ny = (l1_block >> 3);
                __m256i b2[ny], b3[ny];
                for (long l = 0; l < ny; l++) b2[l] = _mm256_sub_epi32(gn, _mm256_loadu_si256((__m256i *)&bkt->pnorm[pnd+l*8]));
                if (record_dp) for (long l = 0; l < ny; l++) b3[l] = _mm256_sub_epi32(cn, _mm256_loadu_si256((__m256i *)&bkt->pdot[pnd+l*8]));

                __m256i upvec_si256[l1_block*nb];
                uint8_t *upvec = (uint8_t *) &upvec_si256[0];
                for (long l = 0; l < l1_block; l += 8) {
                    for (long i = 0; i < nb; i++) {
                        upvec_si256[(l+0)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[pnd+l+0] + i*32)));
                        upvec_si256[(l+1)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[pnd+l+1] + i*32)));
                        upvec_si256[(l+2)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[pnd+l+2] + i*32)));
                        upvec_si256[(l+3)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[pnd+l+3] + i*32)));
                        upvec_si256[(l+4)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[pnd+l+4] + i*32)));
                        upvec_si256[(l+5)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[pnd+l+5] + i*32)));
                        upvec_si256[(l+6)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[pnd+l+6] + i*32)));
                        upvec_si256[(l+7)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[pnd+l+7] + i*32)));
                    }
                }

                long nnd = Nnd;
                while (nnd < Nbound - 7) {
                    __m256i dst[8];
                    uint64_t cmp2[ny] = {}, cmp3[ny] = {};
                    for (long l = 0; l < ny; l++) {
                        vdp8x8(dst, upvec + l * 8 * vec_length, bkt->nvec + nnd, bkt->nsum + nnd);
                        uint64_t cmp2_0 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[nnd+0]), dst[0])));
                        uint64_t cmp2_1 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[nnd+1]), dst[1])));
                        uint64_t cmp2_2 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[nnd+2]), dst[2])));
                        uint64_t cmp2_3 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[nnd+3]), dst[3])));
                        uint64_t cmp2_4 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[nnd+4]), dst[4])));
                        uint64_t cmp2_5 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[nnd+5]), dst[5])));
                        uint64_t cmp2_6 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[nnd+6]), dst[6])));
                        uint64_t cmp2_7 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[nnd+7]), dst[7])));
                        uint64_t cmp2_lo = ( cmp2_0 | (cmp2_1 << 8) | (cmp2_2 << 16) | (cmp2_3 << 24) );
                        uint64_t cmp2_hi = ( cmp2_4 | (cmp2_5 << 8) | (cmp2_6 << 16) | (cmp2_7 << 24) );
                        cmp2[l] = ( cmp2_lo | (cmp2_hi << 32) );
                        if (record_dp) {
                            uint64_t cmp3_0 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[nnd+0]), dst[0])));
                            uint64_t cmp3_1 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[nnd+1]), dst[1])));
                            uint64_t cmp3_2 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[nnd+2]), dst[2])));
                            uint64_t cmp3_3 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[nnd+3]), dst[3])));
                            uint64_t cmp3_4 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[nnd+4]), dst[4])));
                            uint64_t cmp3_5 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[nnd+5]), dst[5])));
                            uint64_t cmp3_6 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[nnd+6]), dst[6])));
                            uint64_t cmp3_7 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[nnd+7]), dst[7])));
                            uint64_t cmp3_lo = ( cmp3_0 | (cmp3_1 << 8) | (cmp3_2 << 16) | (cmp3_3 << 24) );
                            uint64_t cmp3_hi = ( cmp3_4 | (cmp3_5 << 8) | (cmp3_6 << 16) | (cmp3_7 << 24) );
                            cmp3[l] = ( cmp3_lo | (cmp3_hi << 32) );
                        }
                    }
                    
                    for (long l = 0; l < ny; l++) CHECK_AND_ADD_8x8(cmp2[l], (bkt->nvec+nnd), (bkt->pvec+pnd+l*8), TRY_ADDA_TO_DST);
                    if (record_dp) {
                        for (long l = 0; l < ny; l++) CHECK_AND_ADD_8x8(cmp3[l], (bkt->nvec+nnd), (bkt->pvec+pnd+l*8), TRY_ADDAS_TO_DST);
                    }

                    nnd += 8;
                }
                if (nnd < Nbound) {
                    const long nrem = Nbound - nnd;
                    __m256i dst[8];
                    uint64_t cmp2[ny] = {}, cmp3[ny] = {};
                    for (long l = 0; l < ny; l++) {
                        vdp8xn(dst, bkt->nvec+nnd, upvec + l * 8 * vec_length, nrem);
                        for (long i = 0; i < nrem; i++) {
                            dst[i] = _mm256_sub_epi32(dst[i], (__m256i) _mm256_broadcast_ss((float *)&bkt->nsum[nnd+i]));
                        }
                        for (long i = 0; i < nrem; i++) {
                            __m256i lhs2 = _mm256_add_epi32(dst[i], (__m256i)_mm256_broadcast_ss((float *)&bkt->nnorm[nnd+i]));
                            cmp2[l] |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], lhs2))) << (i*8);
                        }
                        if (record_dp) {
                            for (long i = 0; i < nrem; i++) {
                                __m256i lhs3 = _mm256_sub_epi32((__m256i)_mm256_broadcast_ss((float *)&bkt->ndot[nnd+i]), dst[i]);
                                cmp3[l] |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], lhs3))) << (i*8);
                            }
                        }
                    }
                    
                    for (long l = 0; l < ny; l++) CHECK_AND_ADD_8x8(cmp2[l], (bkt->nvec+nnd), (bkt->pvec+pnd+l*8), TRY_ADDA_TO_DST);
                    if (record_dp) {
                        for (long l = 0; l < ny; l++) CHECK_AND_ADD_8x8(cmp3[l], (bkt->nvec+nnd), (bkt->pvec+pnd+l*8), TRY_ADDAS_TO_DST);
                    }
                }
                pnd += l1_block;
            }
            while (pnd < Pbound - 7) {
                __m256i b2, b3; 
                b2 = _mm256_sub_epi32(gn, _mm256_loadu_si256((__m256i *)&bkt->pnorm[pnd]));
                if (record_dp) b3 = _mm256_sub_epi32(cn, _mm256_loadu_si256((__m256i *)&bkt->pdot[pnd]));
                
                __m256i upvec_si256[8*nb];
                uint8_t *upvec = (uint8_t *) &upvec_si256[0];
                for (long i = 0; i < nb; i++) {
                    upvec_si256[0*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[pnd+0] + i*32)));
                    upvec_si256[1*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[pnd+1] + i*32)));
                    upvec_si256[2*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[pnd+2] + i*32)));
                    upvec_si256[3*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[pnd+3] + i*32)));
                    upvec_si256[4*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[pnd+4] + i*32)));
                    upvec_si256[5*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[pnd+5] + i*32)));
                    upvec_si256[6*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[pnd+6] + i*32)));
                    upvec_si256[7*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[pnd+7] + i*32)));
                }

                long nnd = Nnd;
                while (nnd < Nbound - 7) {
                    __m256i dst[8];
                    vdp8x8(dst, upvec, bkt->nvec + nnd, &bkt->nsum[nnd]);

                    uint64_t cmp2 = 0, cmp3 = 0;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[nnd+0]), dst[0])))) << 0;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[nnd+1]), dst[1])))) << 8;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[nnd+2]), dst[2])))) << 16;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[nnd+3]), dst[3])))) << 24;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[nnd+4]), dst[4])))) << 32;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[nnd+5]), dst[5])))) << 40;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[nnd+6]), dst[6])))) << 48;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[nnd+7]), dst[7])))) << 56;
                    if (record_dp) {
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[nnd+0]), dst[0])))) << 0;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[nnd+1]), dst[1])))) << 8;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[nnd+2]), dst[2])))) << 16;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[nnd+3]), dst[3])))) << 24;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[nnd+4]), dst[4])))) << 32;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[nnd+5]), dst[5])))) << 40;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[nnd+6]), dst[6])))) << 48;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[nnd+7]), dst[7])))) << 56;
                    }

                    CHECK_AND_ADD_8x8(cmp2, (bkt->nvec+nnd), (bkt->pvec+pnd), TRY_ADDA_TO_DST);
                    if (record_dp) CHECK_AND_ADD_8x8(cmp3, (bkt->nvec+nnd), (bkt->pvec+pnd), TRY_ADDAS_TO_DST);
                    nnd += 8;
                }
                if (nnd < Nbound) {
                    const long nrem = Nbound - nnd;
                    __m256i dst[8];

                    vdp8xn(dst, bkt->nvec+nnd, upvec, nrem);
                    for (long i = 0; i < nrem; i++) dst[i] = _mm256_sub_epi32(dst[i], (__m256i) _mm256_broadcast_ss((float *)&bkt->nsum[nnd+i]));

                    uint64_t cmp2 = 0, cmp3 = 0;
                    for (long i = 0; i < nrem; i++) {
                        __m256i lhs2 = _mm256_add_epi32(dst[i], (__m256i)_mm256_broadcast_ss((float *)&bkt->nnorm[nnd+i]));
                        cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, lhs2))) << (i*8);
                        if (record_dp) {
                            __m256i lhs3 = _mm256_sub_epi32((__m256i)_mm256_broadcast_ss((float *)&bkt->ndot[nnd+i]), dst[i]);
                            cmp3 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, lhs3))) << (i*8);
                        }
                    }
                    CHECK_AND_ADD_8x8(cmp2, (bkt->nvec+nnd), (bkt->pvec+pnd), TRY_ADDA_TO_DST);
                    if (record_dp) CHECK_AND_ADD_8x8(cmp3, (bkt->nvec+nnd), (bkt->pvec+pnd), TRY_ADDAS_TO_DST);
                }
                pnd += 8;
            }
            // just a small part, we do not change to vertical version
            if (pnd < Pbound) {
                const long prem = Pbound - pnd;
                __m256i b2[7], b3[7];
                for (long i = 0; i < prem; i++) b2[i] = _mm256_sub_epi32(gn, (__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[pnd+i]));
                if (record_dp) {
                    for (long i = 0; i < prem; i++) {
                        b3[i] = _mm256_sub_epi32(cn, (__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[pnd+i]));
                    }
                }
                
                __m256i upvec_si256[7*nb];
                uint8_t *upvec = (uint8_t *) &upvec_si256[0];
                for (long i = 0; i < nb; i++) {
                    for (long j = 0; j < prem; j++) {
                        upvec_si256[j*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[pnd+j] + i*32)));
                    }
                }

                long nnd = Nnd;
                while (nnd < Nbound - 7) {
                    __m256i dst[7];
                    __m256i nsum_si256 = _mm256_loadu_si256((__m256i *)&bkt->nsum[nnd]);
                    __m256i nnorm_si256 = _mm256_loadu_si256((__m256i *)&bkt->nnorm[nnd]);
                    __m256i nnormc_si256;
                    if (record_dp) nnormc_si256 = _mm256_loadu_si256((__m256i *)&bkt->ndot[nnd]);

                    vdpnx8(dst, upvec, bkt->nvec + nnd, prem);
                    for (long i = 0; i < prem; i++) dst[i] = _mm256_sub_epi32(dst[i], nsum_si256);

                    uint64_t cmp2 = 0, cmp3 = 0;
                    for (long i = 0; i < prem; i++) {
                        cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[i], _mm256_add_epi32(nnorm_si256, dst[i])))) << (i*8);
                    }
                    if (record_dp) {
                        for (long i = 0; i < prem; i++) {
                            cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[i], _mm256_sub_epi32(nnormc_si256, dst[i])))) << (i*8);
                        }
                    }

                    CHECK_AND_ADD_8x8(cmp2, (bkt->pvec+pnd), (bkt->nvec+nnd), TRY_ADDA_TO_DST);
                    if (record_dp) CHECK_AND_ADD_8x8(cmp3, (bkt->pvec+pnd), (bkt->nvec+nnd), TRY_ADDSA_TO_DST);
                    nnd += 8;
                }
                if (nnd < Nbound) {
                    const long nrem = Nbound - nnd;
                    int32_t gn_epi32 = goal_norm;
                    int32_t gc_epi32 = goal_norm - bkt->center_norm;
                    for (long i = 0; i < prem; i++) {
                        for (long j = 0; j < nrem; j++) {
                            int32_t dp = vdp(upvec + i * vec_length, vec + bkt->nvec[nnd+j] * vec_length) - bkt->nsum[nnd+j];
                            if (dp + bkt->nnorm[nnd+j] < gn_epi32 - bkt->pnorm[pnd+i]) TRY_ADDA_TO_DST(bkt->pvec[pnd+i], bkt->nvec[nnd+j]);
                            if (record_dp) {
                                if (bkt->ndot[nnd+j] - dp < gc_epi32 - bkt->pdot[pnd+i]) {
                                    TRY_ADDSA_TO_DST(bkt->pvec[pnd+i], bkt->nvec[nnd+j]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (profiling && prof) {
        pthread_spin_lock(&prof->profile_lock);
        prof->try_add2 += try_add2;
        prof->try_add3 += try_add3;
        prof->succ_add2 += succ_add2;
        prof->succ_add3 += succ_add3;
        pthread_spin_unlock(&prof->profile_lock);
    }
    return 1;
}

template <uint32_t nb>
template <uint32_t l2_block, uint32_t l1_block, bool record_dp, bool profiling>
int Pool_epi8_t<nb>::_search_pp(bucket_epi8_t<record_dp> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<nb> *prof) {
    uint64_t try_add2 = 0;
    uint64_t try_add3 = 0;
    uint64_t succ_add2 = 0;
    uint64_t succ_add3 = 0;

    const __m256i gn = _mm256_set1_epi32(goal_norm);
    const __m256i cn = _mm256_set1_epi32(goal_norm - bkt->center_norm);

    for (long Ind = 0; Ind < bkt->num_pvec; Ind += l2_block) {
        for (long Jnd = Ind; Jnd < bkt->num_pvec; Jnd += l2_block) {
            const long Ibound = (Ind + l2_block > bkt->num_pvec) ? bkt->num_pvec : Ind + l2_block;
            const long Jbound = (Jnd + l2_block > bkt->num_pvec) ? bkt->num_pvec : Jnd + l2_block;

            long ind = Ind;
            while (ind < Ibound - l1_block + 1) {
                constexpr long ny = (l1_block >> 3);
                __m256i b2[ny], b3[ny];
                for (long l = 0; l < ny; l++) b2[l] = _mm256_sub_epi32(gn, _mm256_loadu_si256((__m256i *)&bkt->pnorm[ind+l*8]));
                if (record_dp) for (long l = 0; l < ny; l++) b3[l] = _mm256_sub_epi32(cn, _mm256_loadu_si256((__m256i *)&bkt->pdot[ind+l*8]));

                __m256i uivec_si256[l1_block * nb];
                uint8_t *uivec = (uint8_t *) &uivec_si256[0];
                for (long l = 0; l < l1_block; l += 8) {
                    for (long i = 0; i < nb; i++) {
                        uivec_si256[(l+0)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[ind+l+0] + i*32)));
                        uivec_si256[(l+1)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[ind+l+1] + i*32)));
                        uivec_si256[(l+2)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[ind+l+2] + i*32)));
                        uivec_si256[(l+3)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[ind+l+3] + i*32)));
                        uivec_si256[(l+4)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[ind+l+4] + i*32)));
                        uivec_si256[(l+5)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[ind+l+5] + i*32)));
                        uivec_si256[(l+6)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[ind+l+6] + i*32)));
                        uivec_si256[(l+7)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[ind+l+7] + i*32)));
                    }
                }
                
                long jnd = (Ind == Jnd) ? ind + 8 : Jnd;
                if (Ind == Jnd) {
                    // process the trianglar part
                    for (long upp = 1; upp < ny; upp++) {
                        __m256i dst[8];
                        uint64_t cmp2[ny] = {}, cmp3[ny] = {};
                        for (long l = 0; l < upp; l++) {
                            vdp8x8(dst, uivec + l * 8 * vec_length, bkt->pvec + jnd, bkt->psum + jnd);
                            uint64_t cmp2_0 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+0]), dst[0])));
                            uint64_t cmp2_1 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+1]), dst[1])));
                            uint64_t cmp2_2 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+2]), dst[2])));
                            uint64_t cmp2_3 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+3]), dst[3])));
                            uint64_t cmp2_4 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+4]), dst[4])));
                            uint64_t cmp2_5 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+5]), dst[5])));
                            uint64_t cmp2_6 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+6]), dst[6])));
                            uint64_t cmp2_7 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+7]), dst[7])));
                            uint64_t cmp2_lo = ( cmp2_0 | (cmp2_1 << 8) | (cmp2_2 << 16) | (cmp2_3 << 24) );
                            uint64_t cmp2_hi = ( cmp2_4 | (cmp2_5 << 8) | (cmp2_6 << 16) | (cmp2_7 << 24) );
                            cmp2[l] = ( cmp2_lo | (cmp2_hi << 32) );
                            if (record_dp) {
                                uint64_t cmp3_0 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+0]), dst[0])));
                                uint64_t cmp3_1 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+1]), dst[1])));
                                uint64_t cmp3_2 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+2]), dst[2])));
                                uint64_t cmp3_3 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+3]), dst[3])));
                                uint64_t cmp3_4 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+4]), dst[4])));
                                uint64_t cmp3_5 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+5]), dst[5])));
                                uint64_t cmp3_6 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+6]), dst[6])));
                                uint64_t cmp3_7 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+7]), dst[7])));
                                uint64_t cmp3_lo = ( cmp3_0 | (cmp3_1 << 8) | (cmp3_2 << 16) | (cmp3_3 << 24) );
                                uint64_t cmp3_hi = ( cmp3_4 | (cmp3_5 << 8) | (cmp3_6 << 16) | (cmp3_7 << 24) );
                                cmp3[l] = ( cmp3_lo | (cmp3_hi << 32) );
                            }
                        }
                        
                        for (long l = 0; l < upp; l++) CHECK_AND_ADD_8x8(cmp2[l], (bkt->pvec+jnd), (bkt->pvec+ind+l*8), TRY_ADDS_TO_DST);
                        if (record_dp) {
                            for (long l = 0; l < upp; l++) CHECK_AND_ADD_8x8(cmp3[l], (bkt->pvec+jnd), (bkt->pvec+ind+l*8), TRY_ADDSS_TO_DST);
                        }

                        jnd += 8;
                    }
                }
                while (jnd < Jbound - 7) {
                    __m256i dst[8];
                    uint64_t cmp2[ny] = {}, cmp3[ny] = {};
                    for (long l = 0; l < ny; l++) {
                        vdp8x8(dst, uivec + l * 8 * vec_length, bkt->pvec + jnd, bkt->psum + jnd);
                        uint64_t cmp2_0 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+0]), dst[0])));
                        uint64_t cmp2_1 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+1]), dst[1])));
                        uint64_t cmp2_2 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+2]), dst[2])));
                        uint64_t cmp2_3 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+3]), dst[3])));
                        uint64_t cmp2_4 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+4]), dst[4])));
                        uint64_t cmp2_5 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+5]), dst[5])));
                        uint64_t cmp2_6 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+6]), dst[6])));
                        uint64_t cmp2_7 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+7]), dst[7])));
                        uint64_t cmp2_lo = ( cmp2_0 | (cmp2_1 << 8) | (cmp2_2 << 16) | (cmp2_3 << 24) );
                        uint64_t cmp2_hi = ( cmp2_4 | (cmp2_5 << 8) | (cmp2_6 << 16) | (cmp2_7 << 24) );
                        cmp2[l] = ( cmp2_lo | (cmp2_hi << 32) );
                        if (record_dp) {
                            uint64_t cmp3_0 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+0]), dst[0])));
                            uint64_t cmp3_1 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+1]), dst[1])));
                            uint64_t cmp3_2 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+2]), dst[2])));
                            uint64_t cmp3_3 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+3]), dst[3])));
                            uint64_t cmp3_4 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+4]), dst[4])));
                            uint64_t cmp3_5 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+5]), dst[5])));
                            uint64_t cmp3_6 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+6]), dst[6])));
                            uint64_t cmp3_7 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+7]), dst[7])));
                            uint64_t cmp3_lo = ( cmp3_0 | (cmp3_1 << 8) | (cmp3_2 << 16) | (cmp3_3 << 24) );
                            uint64_t cmp3_hi = ( cmp3_4 | (cmp3_5 << 8) | (cmp3_6 << 16) | (cmp3_7 << 24) );
                            cmp3[l] = ( cmp3_lo | (cmp3_hi << 32) );
                        }
                    }
                    
                    for (long l = 0; l < ny; l++) CHECK_AND_ADD_8x8(cmp2[l], (bkt->pvec+jnd), (bkt->pvec+ind+l*8), TRY_ADDS_TO_DST);
                    if (record_dp) {
                        for (long l = 0; l < ny; l++) CHECK_AND_ADD_8x8(cmp3[l], (bkt->pvec+jnd), (bkt->pvec+ind+l*8), TRY_ADDSS_TO_DST);
                    }

                    jnd += 8;
                }
                if (jnd < Jbound) {
                    const long jrem = Jbound - jnd;
                    __m256i dst[8];
                    uint64_t cmp2[ny] = {}, cmp3[ny] = {};

                    for (long l = 0; l < ny; l++) {
                        vdp8xn(dst, bkt->pvec+jnd, uivec + l * 8 * vec_length, jrem);
                        for (long i = 0; i < jrem; i++) {
                            dst[i] = _mm256_sub_epi32(dst[i], (__m256i) _mm256_broadcast_ss((float *)&bkt->psum[jnd+i]));
                        }
                        for (long i = 0; i < jrem; i++) {
                            __m256i lhs2 = _mm256_sub_epi32((__m256i)_mm256_broadcast_ss((float *)&bkt->pnorm[jnd+i]), dst[i]);
                            cmp2[l] |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], lhs2))) << (i*8);
                        }
                        if (record_dp) {
                            for (long i = 0; i < jrem; i++) {
                                __m256i lhs3 = _mm256_add_epi32((__m256i)_mm256_broadcast_ss((float *)&bkt->pdot[jnd+i]), dst[i]);
                                cmp3[l] |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], lhs3))) << (i*8);
                            }
                        }
                    }

                    for (long l = 0; l < ny; l++) CHECK_AND_ADD_8x8(cmp2[l], (bkt->pvec+jnd), (bkt->pvec+ind+l*8), TRY_ADDS_TO_DST);
                    if (record_dp) {
                        for (long l = 0; l < ny; l++) CHECK_AND_ADD_8x8(cmp3[l], (bkt->pvec+jnd), (bkt->pvec+ind+l*8), TRY_ADDSS_TO_DST);
                    }
                }

                ind += l1_block;
            }
            while (ind < Ibound - 7) {
                __m256i b2, b3; 
                b2 = _mm256_sub_epi32(gn, _mm256_loadu_si256((__m256i *)&bkt->pnorm[ind]));
                if (record_dp) b3 = _mm256_sub_epi32(cn, _mm256_loadu_si256((__m256i *)&bkt->pdot[ind]));

                __m256i uivec_si256[8*nb];
                uint8_t *uivec = (uint8_t *) &uivec_si256[0];
                for (long i = 0; i < nb; i++) {
                    uivec_si256[0*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[ind+0] + i*32)));
                    uivec_si256[1*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[ind+1] + i*32)));
                    uivec_si256[2*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[ind+2] + i*32)));
                    uivec_si256[3*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[ind+3] + i*32)));
                    uivec_si256[4*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[ind+4] + i*32)));
                    uivec_si256[5*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[ind+5] + i*32)));
                    uivec_si256[6*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[ind+6] + i*32)));
                    uivec_si256[7*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[ind+7] + i*32)));
                }

                long jnd = (Ind == Jnd) ? ind + 8: Jnd;
                while (jnd < Jbound - 7) {
                    __m256i dst[8];
                    vdp8x8(dst, uivec, bkt->pvec + jnd, &bkt->psum[jnd]);

                    uint64_t cmp2 = 0, cmp3 = 0;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+0]), dst[0])))) << 0;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+1]), dst[1])))) << 8;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+2]), dst[2])))) << 16;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+3]), dst[3])))) << 24;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+4]), dst[4])))) << 32;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+5]), dst[5])))) << 40;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+6]), dst[6])))) << 48;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[jnd+7]), dst[7])))) << 56;
                    if (record_dp) {
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+0]), dst[0])))) << 0;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+1]), dst[1])))) << 8;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+2]), dst[2])))) << 16;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+3]), dst[3])))) << 24;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+4]), dst[4])))) << 32;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+5]), dst[5])))) << 40;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+6]), dst[6])))) << 48;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[jnd+7]), dst[7])))) << 56;
                    }

                    CHECK_AND_ADD_8x8(cmp2, (bkt->pvec+jnd), (bkt->pvec+ind), TRY_ADDS_TO_DST);
                    if (record_dp) CHECK_AND_ADD_8x8(cmp3, (bkt->pvec+jnd), (bkt->pvec+ind), TRY_ADDSS_TO_DST);
                    jnd += 8;
                }
                if (jnd < Jbound) {
                    const long jrem = Jbound - jnd;
                    __m256i dst[8];

                    vdp8xn(dst, bkt->pvec+jnd, uivec, jrem);
                    for (long i = 0; i < jrem; i++) dst[i] = _mm256_sub_epi32(dst[i], (__m256i) _mm256_broadcast_ss((float *)&bkt->psum[jnd+i]));

                    uint64_t cmp2 = 0, cmp3 = 0;
                    for (long i = 0; i < jrem; i++) {
                        __m256i lhs2 = _mm256_sub_epi32((__m256i)_mm256_broadcast_ss((float *)&bkt->pnorm[jnd+i]), dst[i]);
                        cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, lhs2))) << (i*8);
                        if (record_dp) {
                            __m256i lhs3 = _mm256_add_epi32((__m256i)_mm256_broadcast_ss((float *)&bkt->pdot[jnd+i]), dst[i]);
                            cmp3 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, lhs3))) << (i*8);
                        }
                    }
                    CHECK_AND_ADD_8x8(cmp2, (bkt->pvec+jnd), (bkt->pvec+ind), TRY_ADDS_TO_DST);
                    if (record_dp) CHECK_AND_ADD_8x8(cmp3, (bkt->pvec+jnd), (bkt->pvec+ind), TRY_ADDSS_TO_DST);
                }
                ind += 8;
            }
            // just a small part, we do not change to vertical version
            if (ind < Ibound) {
                const long irem = Ibound - ind;
                __m256i b2[7], b3[7];
                for (long i = 0; i < irem; i++) b2[i] = _mm256_sub_epi32(gn, (__m256i) _mm256_broadcast_ss((float *)&bkt->pnorm[ind+i]));
                if (record_dp) {
                    for (long i = 0; i < irem; i++) {
                        b3[i] = _mm256_sub_epi32(cn, (__m256i) _mm256_broadcast_ss((float *)&bkt->pdot[ind+i]));
                    }
                }
                
                __m256i uivec_si256[7*nb];
                uint8_t *uivec = (uint8_t *) &uivec_si256[0];
                for (long i = 0; i < nb; i++) {
                    for (long j = 0; j < irem; j++) {
                        uivec_si256[j*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->pvec[ind+j] + i*32)));
                    }
                }
                
                long jnd = (Ind == Jnd) ? ind + 8 : Jnd;
                while (jnd < Jbound - 7) {
                    __m256i dst[7];
                    __m256i jsum_si256 = _mm256_loadu_si256((__m256i *)&bkt->psum[jnd]);
                    __m256i jnorm_si256 = _mm256_loadu_si256((__m256i *)&bkt->pnorm[jnd]);
                    __m256i jnormc_si256 = _mm256_loadu_si256((__m256i *)&bkt->pdot[jnd]);

                    vdpnx8(dst, uivec, bkt->pvec + jnd, irem);
                    for (long i = 0; i < irem; i++) dst[i] = _mm256_sub_epi32(dst[i], jsum_si256);

                    uint64_t cmp2 = 0, cmp3 = 0;
                    for (long i = 0; i < irem; i++) {
                        cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[i], _mm256_sub_epi32(jnorm_si256, dst[i])))) << (i*8);
                    }
                    if (record_dp) {
                        for (long i = 0; i < irem; i++) {
                            cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[i], _mm256_add_epi32(jnormc_si256, dst[i])))) << (i*8);
                        }
                    }

                    CHECK_AND_ADD_8x8(cmp2, (bkt->pvec+ind), (bkt->pvec+jnd), TRY_ADDS_TO_DST);
                    if (record_dp) CHECK_AND_ADD_8x8(cmp3, (bkt->pvec+ind), (bkt->pvec+jnd), TRY_ADDSS_TO_DST);
                    jnd += 8;
                }
            }
        }
    }

    if (profiling && prof) {
        pthread_spin_lock(&prof->profile_lock);
        prof->try_add2 += try_add2;
        prof->try_add3 += try_add3;
        prof->succ_add2 += succ_add2;
        prof->succ_add3 += succ_add3;
        pthread_spin_unlock(&prof->profile_lock);
    }
    return 1;
}

template <uint32_t nb>
template <uint32_t l2_block, uint32_t l1_block, bool record_dp, bool profiling>
int Pool_epi8_t<nb>::_search_nn(bucket_epi8_t<record_dp> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<nb> *prof) {
    uint64_t try_add2 = 0;
    uint64_t try_add3 = 0;
    uint64_t succ_add2 = 0;
    uint64_t succ_add3 = 0;

    const __m256i gn = _mm256_set1_epi32(goal_norm);
    const __m256i cn = _mm256_set1_epi32(goal_norm - bkt->center_norm);

    for (long Ind = 0; Ind < bkt->num_nvec; Ind += l2_block) {
        for (long Jnd = Ind; Jnd < bkt->num_nvec; Jnd += l2_block) {
            const long Ibound = (Ind + l2_block > bkt->num_nvec) ? bkt->num_nvec : Ind + l2_block;
            const long Jbound = (Jnd + l2_block > bkt->num_nvec) ? bkt->num_nvec : Jnd + l2_block;

            long ind = Ind;
            while (ind < Ibound - l1_block + 1) {
                constexpr long ny = (l1_block >> 3);
                __m256i b2[ny], b3[ny];
                for (long l = 0; l < ny; l++) b2[l] = _mm256_sub_epi32(gn, _mm256_loadu_si256((__m256i *)&bkt->nnorm[ind+l*8]));
                if (record_dp) for (long l = 0; l < ny; l++) b3[l] = _mm256_sub_epi32(cn, _mm256_loadu_si256((__m256i *)&bkt->ndot[ind+l*8]));

                __m256i uivec_si256[l1_block * nb];
                uint8_t *uivec = (uint8_t *) &uivec_si256[0];
                for (long l = 0; l < l1_block; l += 8) {
                    for (long i = 0; i < nb; i++) {
                        uivec_si256[(l+0)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->nvec[ind+l+0] + i*32)));
                        uivec_si256[(l+1)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->nvec[ind+l+1] + i*32)));
                        uivec_si256[(l+2)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->nvec[ind+l+2] + i*32)));
                        uivec_si256[(l+3)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->nvec[ind+l+3] + i*32)));
                        uivec_si256[(l+4)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->nvec[ind+l+4] + i*32)));
                        uivec_si256[(l+5)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->nvec[ind+l+5] + i*32)));
                        uivec_si256[(l+6)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->nvec[ind+l+6] + i*32)));
                        uivec_si256[(l+7)*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->nvec[ind+l+7] + i*32)));
                    }
                }
                
                long jnd = (Ind == Jnd) ? ind + 8 : Jnd;
                if (Ind == Jnd) {
                    // process the trianglar part
                    for (long upp = 1; upp < ny; upp++) {
                        __m256i dst[8];
                        uint64_t cmp2[ny] = {}, cmp3[ny] = {};
                        for (long l = 0; l < upp; l++) {
                            vdp8x8(dst, uivec + l * 8 * vec_length, bkt->nvec + jnd, bkt->nsum + jnd);
                            uint64_t cmp2_0 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+0]), dst[0])));
                            uint64_t cmp2_1 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+1]), dst[1])));
                            uint64_t cmp2_2 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+2]), dst[2])));
                            uint64_t cmp2_3 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+3]), dst[3])));
                            uint64_t cmp2_4 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+4]), dst[4])));
                            uint64_t cmp2_5 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+5]), dst[5])));
                            uint64_t cmp2_6 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+6]), dst[6])));
                            uint64_t cmp2_7 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+7]), dst[7])));
                            uint64_t cmp2_lo = ( cmp2_0 | (cmp2_1 << 8) | (cmp2_2 << 16) | (cmp2_3 << 24) );
                            uint64_t cmp2_hi = ( cmp2_4 | (cmp2_5 << 8) | (cmp2_6 << 16) | (cmp2_7 << 24) );
                            cmp2[l] = ( cmp2_lo | (cmp2_hi << 32) );
                            if (record_dp) {
                                uint64_t cmp3_0 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+0]), dst[0])));
                                uint64_t cmp3_1 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+1]), dst[1])));
                                uint64_t cmp3_2 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+2]), dst[2])));
                                uint64_t cmp3_3 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+3]), dst[3])));
                                uint64_t cmp3_4 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+4]), dst[4])));
                                uint64_t cmp3_5 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+5]), dst[5])));
                                uint64_t cmp3_6 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+6]), dst[6])));
                                uint64_t cmp3_7 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+7]), dst[7])));
                                uint64_t cmp3_lo = ( cmp3_0 | (cmp3_1 << 8) | (cmp3_2 << 16) | (cmp3_3 << 24) );
                                uint64_t cmp3_hi = ( cmp3_4 | (cmp3_5 << 8) | (cmp3_6 << 16) | (cmp3_7 << 24) );
                                cmp3[l] = ( cmp3_lo | (cmp3_hi << 32) );
                            }
                        }
                        
                        for (long l = 0; l < upp; l++) CHECK_AND_ADD_8x8(cmp2[l], (bkt->nvec+jnd), (bkt->nvec+ind+l*8), TRY_ADDS_TO_DST);
                        if (record_dp) {
                            for (long l = 0; l < upp; l++) CHECK_AND_ADD_8x8(cmp3[l], (bkt->nvec+jnd), (bkt->nvec+ind+l*8), TRY_ADDAA_TO_DST);
                        }

                        jnd += 8;
                    }
                }
                while (jnd < Jbound - 7) {
                    __m256i dst[8];
                    uint64_t cmp2[ny] = {}, cmp3[ny] = {};
                    for (long l = 0; l < ny; l++) {
                        vdp8x8(dst, uivec + l * 8 * vec_length, bkt->nvec + jnd, bkt->nsum + jnd);
                        uint64_t cmp2_0 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+0]), dst[0])));
                        uint64_t cmp2_1 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+1]), dst[1])));
                        uint64_t cmp2_2 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+2]), dst[2])));
                        uint64_t cmp2_3 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+3]), dst[3])));
                        uint64_t cmp2_4 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+4]), dst[4])));
                        uint64_t cmp2_5 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+5]), dst[5])));
                        uint64_t cmp2_6 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+6]), dst[6])));
                        uint64_t cmp2_7 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+7]), dst[7])));
                        uint64_t cmp2_lo = ( cmp2_0 | (cmp2_1 << 8) | (cmp2_2 << 16) | (cmp2_3 << 24) );
                        uint64_t cmp2_hi = ( cmp2_4 | (cmp2_5 << 8) | (cmp2_6 << 16) | (cmp2_7 << 24) );
                        cmp2[l] = ( cmp2_lo | (cmp2_hi << 32) );
                        if (record_dp) {
                            uint64_t cmp3_0 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+0]), dst[0])));
                            uint64_t cmp3_1 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+1]), dst[1])));
                            uint64_t cmp3_2 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+2]), dst[2])));
                            uint64_t cmp3_3 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+3]), dst[3])));
                            uint64_t cmp3_4 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+4]), dst[4])));
                            uint64_t cmp3_5 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+5]), dst[5])));
                            uint64_t cmp3_6 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+6]), dst[6])));
                            uint64_t cmp3_7 = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+7]), dst[7])));
                            uint64_t cmp3_lo = ( cmp3_0 | (cmp3_1 << 8) | (cmp3_2 << 16) | (cmp3_3 << 24) );
                            uint64_t cmp3_hi = ( cmp3_4 | (cmp3_5 << 8) | (cmp3_6 << 16) | (cmp3_7 << 24) );
                            cmp3[l] = ( cmp3_lo | (cmp3_hi << 32) );
                        }
                    }
                    
                    for (long l = 0; l < ny; l++) CHECK_AND_ADD_8x8(cmp2[l], (bkt->nvec+jnd), (bkt->nvec+ind+l*8), TRY_ADDS_TO_DST);
                    if (record_dp) {
                        for (long l = 0; l < ny; l++) CHECK_AND_ADD_8x8(cmp3[l], (bkt->nvec+jnd), (bkt->nvec+ind+l*8), TRY_ADDAA_TO_DST);
                    }

                    jnd += 8;
                }
                if (jnd < Jbound) {
                    const long jrem = Jbound - jnd;
                    __m256i dst[8];
                    uint64_t cmp2[ny] = {}, cmp3[ny] = {};

                    for (long l = 0; l < ny; l++) {
                        vdp8xn(dst, bkt->nvec+jnd, uivec + l * 8 * vec_length, jrem);
                        for (long i = 0; i < jrem; i++) {
                            dst[i] = _mm256_sub_epi32(dst[i], (__m256i) _mm256_broadcast_ss((float *)&bkt->nsum[jnd+i]));
                        }
                        for (long i = 0; i < jrem; i++) {
                            __m256i lhs2 = _mm256_sub_epi32((__m256i)_mm256_broadcast_ss((float *)&bkt->nnorm[jnd+i]), dst[i]);
                            cmp2[l] |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[l], lhs2))) << (i*8);
                        }
                        if (record_dp) {
                            for (long i = 0; i < jrem; i++) {
                                __m256i lhs3 = _mm256_add_epi32((__m256i)_mm256_broadcast_ss((float *)&bkt->ndot[jnd+i]), dst[i]);
                                cmp3[l] |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[l], lhs3))) << (i*8);
                            }
                        }
                    }

                    for (long l = 0; l < ny; l++) CHECK_AND_ADD_8x8(cmp2[l], (bkt->nvec+jnd), (bkt->nvec+ind+l*8), TRY_ADDS_TO_DST);
                    if (record_dp) {
                        for (long l = 0; l < ny; l++) CHECK_AND_ADD_8x8(cmp3[l], (bkt->nvec+jnd), (bkt->nvec+ind+l*8), TRY_ADDAA_TO_DST);
                    }
                }

                ind += l1_block;
            }
            while (ind < Ibound - 7) {
                __m256i b2, b3;
                b2 = _mm256_sub_epi32(gn, _mm256_loadu_si256((__m256i *)&bkt->nnorm[ind]));
                if (record_dp) b3 = _mm256_sub_epi32(cn, _mm256_loadu_si256((__m256i *)&bkt->ndot[ind]));

                __m256i uivec_si256[8*nb];
                uint8_t *uivec = (uint8_t *) &uivec_si256[0];
                for (long i = 0; i < nb; i++) {
                    uivec_si256[0*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->nvec[ind+0] + i*32)));
                    uivec_si256[1*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->nvec[ind+1] + i*32)));
                    uivec_si256[2*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->nvec[ind+2] + i*32)));
                    uivec_si256[3*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->nvec[ind+3] + i*32)));
                    uivec_si256[4*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->nvec[ind+4] + i*32)));
                    uivec_si256[5*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->nvec[ind+5] + i*32)));
                    uivec_si256[6*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->nvec[ind+6] + i*32)));
                    uivec_si256[7*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->nvec[ind+7] + i*32)));
                }

                long jnd = (Ind == Jnd) ? ind + 8: Jnd;
                while (jnd < Jbound - 7) {
                    __m256i dst[8];
                    vdp8x8(dst, uivec, bkt->nvec + jnd, &bkt->nsum[jnd]);

                    uint64_t cmp2 = 0, cmp3 = 0;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+0]), dst[0])))) << 0;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+1]), dst[1])))) << 8;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+2]), dst[2])))) << 16;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+3]), dst[3])))) << 24;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+4]), dst[4])))) << 32;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+5]), dst[5])))) << 40;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+6]), dst[6])))) << 48;
                    cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, _mm256_sub_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[jnd+7]), dst[7])))) << 56;
                    if (record_dp) {
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+0]), dst[0])))) << 0;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+1]), dst[1])))) << 8;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+2]), dst[2])))) << 16;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+3]), dst[3])))) << 24;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+4]), dst[4])))) << 32;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+5]), dst[5])))) << 40;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+6]), dst[6])))) << 48;
                        cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, _mm256_add_epi32((__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[jnd+7]), dst[7])))) << 56;
                    }

                    CHECK_AND_ADD_8x8(cmp2, (bkt->nvec+jnd), (bkt->nvec+ind), TRY_ADDS_TO_DST);
                    if (record_dp) CHECK_AND_ADD_8x8(cmp3, (bkt->nvec+jnd), (bkt->nvec+ind), TRY_ADDAA_TO_DST);
                    jnd += 8;
                }
                if (jnd < Jbound) {
                    const long jrem = Jbound - jnd;
                    __m256i dst[8];

                    vdp8xn(dst, bkt->nvec+jnd, uivec, jrem);
                    for (long i = 0; i < jrem; i++) dst[i] = _mm256_sub_epi32(dst[i], (__m256i) _mm256_broadcast_ss((float *)&bkt->nsum[jnd+i]));

                    uint64_t cmp2 = 0, cmp3 = 0;
                    for (long i = 0; i < jrem; i++) {
                        __m256i lhs2 = _mm256_sub_epi32((__m256i)_mm256_broadcast_ss((float *)&bkt->nnorm[jnd+i]), dst[i]);
                        cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2, lhs2))) << (i*8);
                        if (record_dp) {
                            __m256i lhs3 = _mm256_add_epi32((__m256i)_mm256_broadcast_ss((float *)&bkt->ndot[jnd+i]), dst[i]);
                            cmp3 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3, lhs3))) << (i*8);
                        }
                    }
                    CHECK_AND_ADD_8x8(cmp2, (bkt->nvec+jnd), (bkt->nvec+ind), TRY_ADDS_TO_DST);
                    if (record_dp) CHECK_AND_ADD_8x8(cmp3, (bkt->nvec+jnd), (bkt->nvec+ind), TRY_ADDAA_TO_DST);
                }
                ind += 8;
            }
            // just a small part, we do not change to vertical version
            if (ind < Ibound) {
                const long irem = Ibound - ind;
                __m256i b2[7], b3[7];
                for (long i = 0; i < irem; i++) b2[i] = _mm256_sub_epi32(gn, (__m256i) _mm256_broadcast_ss((float *)&bkt->nnorm[ind+i]));
                if (record_dp) {
                    for (long i = 0; i < irem; i++) {
                        b3[i] = _mm256_sub_epi32(cn, (__m256i) _mm256_broadcast_ss((float *)&bkt->ndot[ind+i]));
                    }
                }
                
                __m256i uivec_si256[7*nb];
                uint8_t *uivec = (uint8_t *) &uivec_si256[0];
                for (long i = 0; i < nb; i++) {
                    for (long j = 0; j < irem; j++) {
                        uivec_si256[j*nb+i] = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + vec_length * bkt->nvec[ind+j] + i*32)));
                    }
                }
                
                long jnd = (Ind == Jnd) ? ind + 8 : Jnd;
                while (jnd < Jbound - 7) {
                    __m256i dst[7];
                    __m256i jsum_si256 = _mm256_loadu_si256((__m256i *)&bkt->nsum[jnd]);
                    __m256i jnorm_si256 = _mm256_loadu_si256((__m256i *)&bkt->nnorm[jnd]);
                    __m256i jnormc_si256 = _mm256_loadu_si256((__m256i *)&bkt->ndot[jnd]);

                    vdpnx8(dst, uivec, bkt->nvec + jnd, irem);
                    for (long i = 0; i < irem; i++) dst[i] = _mm256_sub_epi32(dst[i], jsum_si256);

                    uint64_t cmp2 = 0, cmp3 = 0;
                    for (long i = 0; i < irem; i++) {
                        cmp2 |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b2[i], _mm256_sub_epi32(jnorm_si256, dst[i])))) << (i*8);
                    }
                    if (record_dp) {
                        for (long i = 0; i < irem; i++) {
                            cmp3 |= ((uint64_t )_mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(b3[i], _mm256_add_epi32(jnormc_si256, dst[i])))) << (i*8);
                        }
                    }

                    CHECK_AND_ADD_8x8(cmp2, (bkt->nvec+ind), (bkt->nvec+jnd), TRY_ADDS_TO_DST);
                    if (record_dp) CHECK_AND_ADD_8x8(cmp3, (bkt->nvec+ind), (bkt->nvec+jnd), TRY_ADDSS_TO_DST);
                    jnd += 8;
                }
            }
        }
    }

    if (profiling && prof) {
        pthread_spin_lock(&prof->profile_lock);
        prof->try_add2 += try_add2;
        prof->try_add3 += try_add3;
        prof->succ_add2 += succ_add2;
        prof->succ_add3 += succ_add3;
        pthread_spin_unlock(&prof->profile_lock);
    }
    return 1;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::_sol_list_to_vec(sol_list_epi8_t **sol_list, long num_sol_list, int8_t *dst_vec, uint64_t *dst_vu, int32_t *dst_vnorm, int32_t *dst_vsum) {
    long *begin_ind = (long *) NEW_VEC(num_sol_list, sizeof(long));
    long *end_ind = (long *) NEW_VEC(num_sol_list, sizeof(long));
    end_ind[0] = sol_list[0]->num_sol();
    for (long i = 1; i < num_sol_list; i++) {
        begin_ind[i] = end_ind[i-1];
        end_ind[i] = begin_ind[i] + sol_list[i]->num_sol();
    }

    #pragma omp parallel for
    for (long thread = 0; thread < num_sol_list; thread++) {
        __attribute__ ((aligned (32))) int8_t tmp[vec_length * 8];
        __attribute__ ((aligned (32))) int16_t tck[vec_length * 8];
        __attribute__ ((aligned (32))) int32_t coeff[8 * vec_length];
        __attribute__ ((aligned (32))) float fvec[vec_length * 8];
        __attribute__ ((aligned (32))) float fnorm[8];
        __attribute__ ((aligned (32))) int32_t sum[8];

        const __m256i diff_bound = _mm256_set1_epi16(0x3);

        long ind = begin_ind[thread];
        long status, status_ind;
        sol_list[thread]->init(status, status_ind);
        while (ind < end_ind[thread] - 7) {
            for (long i = 0; i < 8; i++) {
                if (status == 0) {
                    int8_t *src1 = vec + sol_list[thread]->a_list[2*status_ind] * vec_length;
                    int8_t *src2 = vec + sol_list[thread]->a_list[2*status_ind+1] * vec_length;
                    copy_avx2(tmp + i * vec_length, src1, vec_length);
                    add_avx2(tmp + i * vec_length, src2, vec_length);
                    for (long l = 0; l < vec_length; l += 16) {
                        __m256i x1 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src1 + l)));
                        __m256i x2 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src2 + l)));
                        _mm256_store_si256((__m256i *)(tck + i * vec_length + l), _mm256_add_epi16(x1, x2));
                    }
                    dst_vu[ind+i] = vu[sol_list[thread]->a_list[2*status_ind]] + vu[sol_list[thread]->a_list[2*status_ind+1]];
                } else if (status == 1) {
                    int8_t *src1 = vec + sol_list[thread]->s_list[2*status_ind] * vec_length;
                    int8_t *src2 = vec + sol_list[thread]->s_list[2*status_ind+1] * vec_length;
                    copy_avx2(tmp + i * vec_length, src1, vec_length);
                    sub_avx2(tmp + i * vec_length, src2, vec_length);
                    for (long l = 0; l < vec_length; l += 16) {
                        __m256i x1 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src1 + l)));
                        __m256i x2 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src2 + l)));
                        _mm256_store_si256((__m256i *)(tck + i * vec_length + l), _mm256_sub_epi16(x1, x2));
                    }
                    dst_vu[ind+i] = vu[sol_list[thread]->s_list[2*status_ind]] - vu[sol_list[thread]->s_list[2*status_ind+1]];
                } else if (status == 2) {
                    int8_t *src1 = vec + sol_list[thread]->aa_list[3*status_ind] * vec_length;
                    int8_t *src2 = vec + sol_list[thread]->aa_list[3*status_ind+1] * vec_length;
                    int8_t *src3 = vec + sol_list[thread]->aa_list[3*status_ind+2] * vec_length;
                    copy_avx2(tmp + i * vec_length, src1, vec_length);
                    add_avx2(tmp + i * vec_length, src2, vec_length);
                    add_avx2(tmp + i * vec_length, src3, vec_length);
                    for (long l = 0; l < vec_length; l += 16) {
                        __m256i x1 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src1 + l)));
                        __m256i x2 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src2 + l)));
                        __m256i x3 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src3 + l)));
                        _mm256_store_si256((__m256i *)(tck + i * vec_length + l), _mm256_add_epi16(_mm256_add_epi16(x1, x2), x3));
                    }
                    dst_vu[ind+i] = vu[sol_list[thread]->aa_list[3*status_ind]] + 
                            vu[sol_list[thread]->aa_list[3*status_ind+1]] + 
                            vu[sol_list[thread]->aa_list[3*status_ind+2]];
                } else if (status == 3) {
                    int8_t *src1 = vec + sol_list[thread]->sa_list[3*status_ind] * vec_length;
                    int8_t *src2 = vec + sol_list[thread]->sa_list[3*status_ind+1] * vec_length;
                    int8_t *src3 = vec + sol_list[thread]->sa_list[3*status_ind+2] * vec_length;
                    copy_avx2(tmp + i * vec_length, src1, vec_length);
                    sub_avx2(tmp + i * vec_length, src2, vec_length);
                    add_avx2(tmp + i * vec_length, src3, vec_length);
                    for (long l = 0; l < vec_length; l += 16) {
                        __m256i x1 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src1 + l)));
                        __m256i x2 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src2 + l)));
                        __m256i x3 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src3 + l)));
                        _mm256_store_si256((__m256i *)(tck + i * vec_length + l), _mm256_add_epi16(_mm256_sub_epi16(x1, x2), x3));
                    }
                    dst_vu[ind+i] = vu[sol_list[thread]->sa_list[3*status_ind]] - 
                            vu[sol_list[thread]->sa_list[3*status_ind+1]] + 
                            vu[sol_list[thread]->sa_list[3*status_ind+2]];
                } else if (status == 4) {
                    int8_t *src1 = vec + sol_list[thread]->ss_list[3*status_ind] * vec_length;
                    int8_t *src2 = vec + sol_list[thread]->ss_list[3*status_ind+1] * vec_length;
                    int8_t *src3 = vec + sol_list[thread]->ss_list[3*status_ind+2] * vec_length;
                    copy_avx2(tmp + i * vec_length, src1, vec_length);
                    sub_avx2(tmp + i * vec_length, src2, vec_length);
                    sub_avx2(tmp + i * vec_length, src3, vec_length);
                    for (long l = 0; l < vec_length; l += 16) {
                        __m256i x1 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src1 + l)));
                        __m256i x2 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src2 + l)));
                        __m256i x3 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src3 + l)));
                        _mm256_store_si256((__m256i *)(tck + i * vec_length + l), _mm256_sub_epi16(_mm256_sub_epi16(x1, x2), x3));
                    }
                    dst_vu[ind+i] = vu[sol_list[thread]->ss_list[3*status_ind]] - 
                            vu[sol_list[thread]->ss_list[3*status_ind+1]] - 
                            vu[sol_list[thread]->ss_list[3*status_ind+2]];
                }
                sol_list[thread]->next(status, status_ind);
            }

            _compute_sum_b8(sum, tmp);
            _compute_coeff_b8(coeff, tmp, sum);
            _compute_fvec_b8(fvec, coeff);
            _compute_fnorm_b8(fnorm, fvec);
            _mm256_storeu_si256((__m256i *) (dst_vnorm + ind), _mm256_cvtps_epi32(_mm256_load_ps(fnorm)));            
            
            for (long i = 0; i < vec_length; i += 16) {
                for (long j = 0; j < 8; j++) {
                    __m128i lo = _mm256_cvtepi32_epi8(_mm256_cvtps_epi32(_mm256_load_ps(fvec + j * vec_length + i)));
                    __m128i hi = _mm256_cvtepi32_epi8(_mm256_cvtps_epi32(_mm256_load_ps(fvec + j * vec_length + i + 8)));
                    _mm_store_si128((__m128i *)(dst_vec + (ind + j) * vec_length + i), _mm_or_si128(_mm_slli_si128(hi, 8), lo));
                }                
            }


            uint32_t rej = 0;
            for (long i = 0; i < 8; i++) {
                for (long l = 0; l < vec_length; l += 16) {
                    __m256i diff = _mm256_abs_epi16(_mm256_sub_epi16(_mm256_load_si256((__m256i *)(tck + i * vec_length + l)), _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(dst_vec + (ind + i) * vec_length + l)))));
                    if (_mm256_movemask_epi8(_mm256_cmpgt_epi16(diff, diff_bound))) {
                        rej |= (1 << i);
                        break;
                    }
                }
                #if REJ_ENTRY128
                do {
                    __m256i err0x80 = _mm256_setzero_si256();
                    __m256i all0x80 = _mm256_set1_epi32(0x80808080);
                    for (long l = 0; l < vec_length; l += 32) {
                        err0x80 = _mm256_or_si256(err0x80, _mm256_cmpeq_epi8(all0x80, _mm256_load_si256((__m256i *)(dst_vec + (ind + i) * vec_length + l))));
                    }
                    if (!_mm256_testz_si256(err0x80, err0x80)) rej |= (1 << i);
                } while (0);
                #endif
            }
            while (rej) {
                int32_t r = __builtin_ctz(rej);
                rej -= (1 << r);
                dst_vnorm[ind + r] = 2147483647;
            }

            _compute_sum_b8(&dst_vsum[ind], dst_vec + ind * vec_length);
            
            ind += 8;
        }

        while (ind < end_ind[thread]) {
            if (status == 0) {
                int8_t *src1 = vec + sol_list[thread]->a_list[2*status_ind] * vec_length;
                int8_t *src2 = vec + sol_list[thread]->a_list[2*status_ind+1] * vec_length;
                copy_avx2(tmp, src1, vec_length);
                add_avx2(tmp, src2, vec_length);
                for (long l = 0; l < vec_length; l += 16) {
                    __m256i x1 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src1 + l)));
                    __m256i x2 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src2 + l)));
                    _mm256_store_si256((__m256i *)(tck + l), _mm256_add_epi16(x1, x2));
                }
                dst_vu[ind] = vu[sol_list[thread]->a_list[2*status_ind]] + vu[sol_list[thread]->a_list[2*status_ind+1]];
            } else if (status == 1) {
                int8_t *src1 = vec + sol_list[thread]->s_list[2*status_ind] * vec_length;
                int8_t *src2 = vec + sol_list[thread]->s_list[2*status_ind+1] * vec_length;
                copy_avx2(tmp, src1, vec_length);
                sub_avx2(tmp, src2, vec_length);
                for (long l = 0; l < vec_length; l += 16) {
                    __m256i x1 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src1 + l)));
                    __m256i x2 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src2 + l)));
                    _mm256_store_si256((__m256i *)(tck + l), _mm256_sub_epi16(x1, x2));
                }
                dst_vu[ind] = vu[sol_list[thread]->s_list[2*status_ind]] - vu[sol_list[thread]->s_list[2*status_ind+1]];
            } else if (status == 2) {
                int8_t *src1 = vec + sol_list[thread]->aa_list[3*status_ind] * vec_length;
                int8_t *src2 = vec + sol_list[thread]->aa_list[3*status_ind+1] * vec_length;
                int8_t *src3 = vec + sol_list[thread]->aa_list[3*status_ind+2] * vec_length;
                copy_avx2(tmp, src1, vec_length);
                add_avx2(tmp, src2, vec_length);
                add_avx2(tmp, src3, vec_length);
                for (long l = 0; l < vec_length; l += 16) {
                    __m256i x1 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src1 + l)));
                    __m256i x2 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src2 + l)));
                    __m256i x3 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src3 + l)));
                    _mm256_store_si256((__m256i *)(tck + l), _mm256_add_epi16(_mm256_add_epi16(x1, x2), x3));
                }
                dst_vu[ind] = vu[sol_list[thread]->aa_list[3*status_ind]] + 
                            vu[sol_list[thread]->aa_list[3*status_ind+1]] + 
                            vu[sol_list[thread]->aa_list[3*status_ind+2]];
            } else if (status == 3) {
                int8_t *src1 = vec + sol_list[thread]->sa_list[3*status_ind] * vec_length;
                int8_t *src2 = vec + sol_list[thread]->sa_list[3*status_ind+1] * vec_length;
                int8_t *src3 = vec + sol_list[thread]->sa_list[3*status_ind+2] * vec_length;
                copy_avx2(tmp, src1, vec_length);
                sub_avx2(tmp, src2, vec_length);
                add_avx2(tmp, src3, vec_length);
                for (long l = 0; l < vec_length; l += 16) {
                    __m256i x1 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src1 + l)));
                    __m256i x2 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src2 + l)));
                    __m256i x3 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src3 + l)));
                    _mm256_store_si256((__m256i *)(tck + l), _mm256_add_epi16(_mm256_sub_epi16(x1, x2), x3));
                }
                dst_vu[ind] = vu[sol_list[thread]->sa_list[3*status_ind]] - 
                            vu[sol_list[thread]->sa_list[3*status_ind+1]] + 
                            vu[sol_list[thread]->sa_list[3*status_ind+2]];
            } else if (status == 4) {
                int8_t *src1 = vec + sol_list[thread]->ss_list[3*status_ind] * vec_length;
                int8_t *src2 = vec + sol_list[thread]->ss_list[3*status_ind+1] * vec_length;
                int8_t *src3 = vec + sol_list[thread]->ss_list[3*status_ind+2] * vec_length;
                copy_avx2(tmp, src1, vec_length);
                sub_avx2(tmp, src2, vec_length);
                sub_avx2(tmp, src3, vec_length);
                for (long l = 0; l < vec_length; l += 16) {
                    __m256i x1 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src1 + l)));
                    __m256i x2 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src2 + l)));
                    __m256i x3 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src3 + l)));
                    _mm256_store_si256((__m256i *)(tck + l), _mm256_sub_epi16(_mm256_sub_epi16(x1, x2), x3));
                }
                dst_vu[ind] = vu[sol_list[thread]->ss_list[3*status_ind]] - 
                            vu[sol_list[thread]->ss_list[3*status_ind+1]] - 
                            vu[sol_list[thread]->ss_list[3*status_ind+2]];
            }
            sol_list[thread]->next(status, status_ind);
            
            dst_vsum[ind] = _compute_sum(tmp);
            _compute_coeff(coeff, tmp, dst_vsum[ind]);
            _compute_fvec(fvec, coeff);
            fnorm[0] = 0.5 * dot_avx2(fvec, fvec, vec_length);
            dst_vnorm[ind] = round(fnorm[0]);
            for (long i = 0; i < vec_length; i += 16) {
                __m128i lo = _mm256_cvtepi32_epi8(_mm256_cvtps_epi32(_mm256_load_ps(fvec + i)));
                __m128i hi = _mm256_cvtepi32_epi8(_mm256_cvtps_epi32(_mm256_load_ps(fvec + i + 8)));
                _mm_store_si128((__m128i *)(dst_vec + ind * vec_length + i), _mm_or_si128(_mm_slli_si128(hi, 8), lo));          
            }

            for (long l = 0; l < vec_length; l += 16) {
                __m256i diff = _mm256_abs_epi16(_mm256_sub_epi16(_mm256_load_si256((__m256i *)(tck + l)), _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(dst_vec + ind * vec_length + l)))));
                if (_mm256_movemask_epi8(_mm256_cmpgt_epi16(diff, diff_bound))) {
                    dst_vnorm[ind] = 2147483647;
                    break;
                }
            }

            #if REJ_ENTRY128
            do {
                __m256i err0x80 = _mm256_setzero_si256();
                __m256i all0x80 = _mm256_set1_epi32(0x80808080);
                for (long l = 0; l < vec_length; l += 32) {
                    err0x80 = _mm256_or_si256(err0x80, _mm256_cmpeq_epi8(all0x80, _mm256_load_si256((__m256i *)(dst_vec + ind * vec_length + l))));
                }
                if (!_mm256_testz_si256(err0x80, err0x80)) dst_vnorm[ind] = 2147483647;
            } while (0);
            #endif

            dst_vsum[ind] = _compute_sum(dst_vec + ind * vec_length);
            ind++;
        }
    }

    return 0;
}       

template <uint32_t nb>
template <bool profiling>
uint64_t Pool_epi8_t<nb>::_pool_insert(sol_list_epi8_t **sol_list, long num_sol_list, int32_t goal_norm, int32_t goal_index, bgj_profile_data_t<nb> *prof) {
    uint64_t num_total_insert = 0;

    uint64_t length_stat[256] = {};
    uint64_t num_linfty_failed = 0;
    uint64_t num_l2_failed = 0;
    uint64_t num_not_try = 0;

    long num_total_sol = 0;
    long num_total_empty;
    long num_total_nonempty;
    long num_sol[MAX_NTHREADS];
    long num_emptyy[MAX_NTHREADS];
    long num_nonemptyy[MAX_NTHREADS];
    long empty_begin[MAX_NTHREADS];
    long empty_end[MAX_NTHREADS];
    long nonempty_begin[MAX_NTHREADS];
    long nonempty_end[MAX_NTHREADS];

    for (long i = 0; i < num_threads; i++) {
        num_sol[i] = sol_list[i]->num_sol();
        num_total_sol += num_sol[i];
    }
    num_total_empty = ( num_total_sol > num_empty ) ? num_empty : num_total_sol;
    num_total_nonempty = ( num_total_sol - num_total_empty > sorted_index - goal_index) ? (sorted_index - goal_index) : (num_total_sol - num_total_empty);
    for (long i = 0; i < num_threads; i++) {
        num_emptyy[i] = (num_total_empty * (i+1)) / num_threads - (num_total_empty * i) / num_threads;
        num_nonemptyy[i] = (num_total_nonempty * (i+1)) / num_threads - (num_total_nonempty * i) / num_threads;
    }
    empty_begin[0] = num_vec;
    empty_end[0] = num_vec + num_emptyy[0];
    nonempty_begin[0] = sorted_index - num_nonemptyy[0];
    nonempty_end[0] = sorted_index;
    for (long i = 1; i < num_threads; i++) {
        empty_begin[i] = empty_end[i-1];
        empty_end[i] = empty_begin[i] + num_emptyy[i];
        nonempty_end[i] = nonempty_begin[i-1];
        nonempty_begin[i] = nonempty_end[i] - num_nonemptyy[i];
    }

    int8_t *vec_to_insert = (int8_t *) NEW_VEC(num_total_sol * vec_length, sizeof(int8_t));
    uint64_t *vu_to_insert = (uint64_t *) NEW_VEC(num_total_sol, sizeof(uint64_t));
    int32_t *vnorm_to_insert = (int32_t *) NEW_VEC(num_total_sol, sizeof(int32_t));
    int32_t *vsum_to_insert = (int32_t *) NEW_VEC(num_total_sol, sizeof(int32_t));
    _sol_list_to_vec(sol_list, num_sol_list, vec_to_insert, vu_to_insert, vnorm_to_insert, vsum_to_insert);

    long empty_final_ind[MAX_NTHREADS];
    long nonempty_final_ind[MAX_NTHREADS];
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++) {
        const long begin_ind = (num_total_sol * thread) / num_threads;
        const long end_ind = (num_total_sol * (thread+1)) / num_threads;
        long ind = begin_ind;
        long empty_ind = empty_begin[thread];
        long nonempty_ind = nonempty_end[thread] - 1;
        
        uint64_t _length_stat[256] = {};
        uint64_t _num_linfty_failed = 0;
        uint64_t _num_l2_failed = 0;

        const int32_t linfty_fail_bound = 1.2 * goal_norm;

        while (ind < end_ind && nonempty_ind >= nonempty_begin[thread]) {
            if (vnorm_to_insert[ind] > linfty_fail_bound) {
                if (profiling) _num_linfty_failed++;
                if (!uid->erase_uid(vu_to_insert[ind])){
                    fprintf(stderr, "[Error] Pool_epi8_t<%u>::_pool_insert: erase uid of non inserted vector failed, ignored.\n", nb);
                }
                ind++;
                continue;
            }
            
            uint32_t dst = *((uint32_t *)(cvec + 3LL * nonempty_ind));
            if (profiling) {
                int r = round(sqrt((10000.0 * vnorm_to_insert[ind]) / goal_norm));
                if (r > 255) r = 255;
                if (r < 0) r = 0;
                _length_stat[r]++;
            }
            if (vnorm_to_insert[ind] < vnorm[dst]) {
                if (!uid->erase_uid(vu[dst])){
                    fprintf(stderr, "[Error] Pool_epi8_t<%u>::_pool_insert: erase uid of pool vector failed, ignored.\n", nb);
                }
                copy_avx2(vec + dst * vec_length, vec_to_insert + ind * vec_length, vec_length);
                vnorm[dst] = vnorm_to_insert[ind];
                vu[dst] = vu_to_insert[ind];
                vsum[dst] = vsum_to_insert[ind];
                int32_t cnorm = ((vnorm[dst] >> 1) > 65535) ? 65535 : (vnorm[dst] >> 1);
                cvec[nonempty_ind*3LL+2LL] = cnorm; 
                nonempty_ind--;
                ind++;
                continue;
            } else if (empty_ind < empty_end[thread]) {
                copy_avx2(vec + empty_ind * vec_length, vec_to_insert + ind * vec_length, vec_length);
                vnorm[empty_ind] = vnorm_to_insert[ind];
                vsum[empty_ind] = vsum_to_insert[ind];
                vu[empty_ind] = vu_to_insert[ind];
                int32_t cnorm = ((vnorm[empty_ind] >> 1) > 65535) ? 65535 : (vnorm[empty_ind] >> 1);
                cvec[empty_ind * 3LL + 2LL] = cnorm;
                *((uint32_t *) (cvec + empty_ind * 3LL)) = empty_ind;
                ind++;
                empty_ind++;
                continue;
            } else {
                if (profiling) _num_l2_failed++;
                if (!uid->erase_uid(vu_to_insert[ind])){
                    fprintf(stderr, "[Error] Pool_epi8_t<%u>::_pool_insert: erase uid of non inserted vector failed, ignored.\n", nb);
                }
                ind++;
            }
        }

        while (ind < end_ind && empty_ind < empty_end[thread]) {
            if (vnorm_to_insert[ind] > linfty_fail_bound) {
                if (profiling) _num_linfty_failed++;
                if (!uid->erase_uid(vu_to_insert[ind])){
                    fprintf(stderr, "[Error] Pool_epi8_t<%u>::_pool_insert: erase uid of non inserted vector failed, ignored.\n", nb);
                }
                ind++;
                continue;
            }
            if (profiling) {
                int r = round(sqrt((10000.0 * vnorm_to_insert[ind]) / goal_norm));
                if (r > 255) r = 255;
                if (r < 0) r = 0;
                _length_stat[r]++;
            }
            copy_avx2(vec + empty_ind * vec_length, vec_to_insert + ind * vec_length, vec_length);
            vnorm[empty_ind] = vnorm_to_insert[ind];
            vsum[empty_ind] = vsum_to_insert[ind];
            vu[empty_ind] = vu_to_insert[ind];
            int32_t cnorm = ((vnorm[empty_ind] >> 1) > 65535) ? 65535 : (vnorm[empty_ind] >> 1);
            cvec[empty_ind * 3LL + 2LL] = cnorm;
            *((uint32_t *) (cvec + empty_ind * 3LL)) = empty_ind;
            ind++;
            empty_ind++;
        }

        while (ind < end_ind) {
            if (!uid->erase_uid(vu_to_insert[ind])){
                fprintf(stderr, "[Error] Pool_epi8_t<%u>::_pool_insert: erase uid of non inserted vector failed, ignored.\n", nb);
            }
            ind++;
        }
 
        if (prof) {
            uint64_t _num_total_insert = (empty_ind - empty_begin[thread]) + (nonempty_end[thread] - 1 - nonempty_ind);
            uint64_t _num_not_try = num_sol[thread] - _num_total_insert - _num_linfty_failed - _num_l2_failed;
            pthread_spin_lock(&prof->profile_lock);
            if (profiling) {
                for (long i = 0; i < 256; i++) length_stat[i] += _length_stat[i];
                num_linfty_failed += _num_linfty_failed;
                num_l2_failed += _num_l2_failed;
                num_not_try += _num_not_try;
            }
            num_total_insert += _num_total_insert;
            pthread_spin_unlock(&prof->profile_lock);
        }
        empty_final_ind[thread] = empty_ind;
        nonempty_final_ind[thread] = nonempty_ind;
    }
    
    FREE_VEC((void *)vec_to_insert);
    FREE_VEC((void *)vu_to_insert);
    FREE_VEC((void *)vnorm_to_insert);
    FREE_VEC((void *)vsum_to_insert);

    for (long i = 0; i < num_threads - 1; i++) {
        for (long k = num_threads - 1; k > i; k--) {
            while ((empty_final_ind[i] < empty_end[i]) && (empty_final_ind[k] > empty_begin[k])) {
                uint32_t src = empty_final_ind[k] - 1;
                uint32_t dst = empty_final_ind[i];
                memcpy(vec + dst * vec_length, vec + src * vec_length, vec_length);
                vnorm[dst] = vnorm[src];
                vsum[dst] = vsum[src];
                vu[dst] = vu[src];
                cvec[(long)dst*3LL+2LL] = cvec[(long)src*3LL+2LL];
                ((uint32_t *)(cvec + (long)dst * 3LL))[0] = dst;
                empty_final_ind[i]++;
                empty_final_ind[k]--;
            }
        }
    }
    for (long i = 0; i < num_threads; i++) {
        num_vec += empty_final_ind[i] - empty_begin[i];
        num_empty -= empty_final_ind[i] - empty_begin[i];
    }
    sorted_index = nonempty_final_ind[num_threads-1];

    if (profiling && prof) {
        prof->insert_inner_log(length_stat, num_linfty_failed, num_l2_failed, num_not_try);
    }

    return num_total_insert;
}

#if COMPILE_POOL_EPI8_96
template int Pool_epi8_t<3>::_pool_bucketing<BGJ1_EPI8_BUCKET_BATCHSIZE, BGJ1_EPI8_USE_3RED, BGJ1_EPI8_USE_FARAWAY_CENTER, 1, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
template int Pool_epi8_t<3>::_pool_bucketing<BGJ1_EPI8_BUCKET_BATCHSIZE, BGJ1_EPI8_USE_3RED, BGJ1_EPI8_USE_FARAWAY_CENTER, 1, 0>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
template int Pool_epi8_t<3>::_pool_bucketing<BGJ2_EPI8_BUCKET0_BATCHSIZE, BGJ2_EPI8_REUSE_USE_3RED, BGJ2_EPI8_USE_FARAWAY_CENTER, 0, 1>(bucket_epi8_t<BGJ2_EPI8_REUSE_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
template int Pool_epi8_t<3>::_pool_bucketing<BGJ2_EPI8_BUCKET0_BATCHSIZE, BGJ2_EPI8_REUSE_USE_3RED, BGJ2_EPI8_USE_FARAWAY_CENTER, 0, 0>(bucket_epi8_t<BGJ2_EPI8_REUSE_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
#if BGJ3_EPI8_BUCKET0_BATCHSIZE != BGJ2_EPI8_BUCKET0_BATCHSIZE
template int Pool_epi8_t<3>::_pool_bucketing<BGJ3_EPI8_BUCKET0_BATCHSIZE, BGJ3_EPI8_REUSE0_USE_3RED, BGJ3_EPI8_USE_FARAWAY_CENTER, 0, 1>(bucket_epi8_t<BGJ3_EPI8_REUSE0_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
template int Pool_epi8_t<3>::_pool_bucketing<BGJ3_EPI8_BUCKET0_BATCHSIZE, BGJ3_EPI8_REUSE0_USE_3RED, BGJ3_EPI8_USE_FARAWAY_CENTER, 0, 0>(bucket_epi8_t<BGJ3_EPI8_REUSE0_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
#endif

template int Pool_epi8_t<3>::_sub_bucketing<BGJ2_EPI8_BUCKET1_BATCHSIZE, BGJ2_EPI8_USE_FARAWAY_CENTER, 1, 1, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::_sub_bucketing<BGJ2_EPI8_BUCKET1_BATCHSIZE, BGJ2_EPI8_USE_FARAWAY_CENTER, 1, 0, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::_sub_bucketing<BGJ3_EPI8_BUCKET1_BATCHSIZE, BGJ3_EPI8_USE_FARAWAY_CENTER, 0, 1, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::_sub_bucketing<BGJ3_EPI8_BUCKET1_BATCHSIZE, BGJ3_EPI8_USE_FARAWAY_CENTER, 0, 0, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
#if BGJ3_EPI8_BUCKET2_BATCHSIZE != BGJ2_EPI8_BUCKET2_BATCHSIZE
template int Pool_epi8_t<3>::_sub_bucketing<BGJ3_EPI8_BUCKET2_BATCHSIZE, BGJ3_EPI8_USE_FARAWAY_CENTER, 1, 1, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::_sub_bucketing<BGJ3_EPI8_BUCKET2_BATCHSIZE, BGJ3_EPI8_USE_FARAWAY_CENTER, 1, 0, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
#endif

#if BGJ_NEED_SEARCH_3RED
template int Pool_epi8_t<3>::_search_cred<1, 1>(bucket_epi8_t<1> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::_search_np<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 1, 1>(bucket_epi8_t<1> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::_search_pp<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 1, 1>(bucket_epi8_t<1> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::_search_nn<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 1, 1>(bucket_epi8_t<1> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
#endif

template int Pool_epi8_t<3>::_search_cred<0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::_search_np<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::_search_pp<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::_search_nn<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);

template uint64_t Pool_epi8_t<3>::_pool_insert<1>(sol_list_epi8_t **sol_list, long num_sol_list, int32_t goal_norm, int32_t goal_index, bgj_profile_data_t<3> *prof);
template uint64_t Pool_epi8_t<3>::_pool_insert<0>(sol_list_epi8_t **sol_list, long num_sol_list, int32_t goal_norm, int32_t goal_index, bgj_profile_data_t<3> *prof);
#endif

#if COMPILE_POOL_EPI8_128
template int Pool_epi8_t<4>::_pool_bucketing<BGJ1_EPI8_BUCKET_BATCHSIZE, BGJ1_EPI8_USE_3RED, BGJ1_EPI8_USE_FARAWAY_CENTER, 1, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
template int Pool_epi8_t<4>::_pool_bucketing<BGJ1_EPI8_BUCKET_BATCHSIZE, BGJ1_EPI8_USE_3RED, BGJ1_EPI8_USE_FARAWAY_CENTER, 1, 0>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
template int Pool_epi8_t<4>::_pool_bucketing<BGJ2_EPI8_BUCKET0_BATCHSIZE, BGJ2_EPI8_REUSE_USE_3RED, BGJ2_EPI8_USE_FARAWAY_CENTER, 0, 1>(bucket_epi8_t<BGJ2_EPI8_REUSE_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
template int Pool_epi8_t<4>::_pool_bucketing<BGJ2_EPI8_BUCKET0_BATCHSIZE, BGJ2_EPI8_REUSE_USE_3RED, BGJ2_EPI8_USE_FARAWAY_CENTER, 0, 0>(bucket_epi8_t<BGJ2_EPI8_REUSE_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
#if BGJ3_EPI8_BUCKET0_BATCHSIZE != BGJ2_EPI8_BUCKET0_BATCHSIZE
template int Pool_epi8_t<4>::_pool_bucketing<BGJ3_EPI8_BUCKET0_BATCHSIZE, BGJ3_EPI8_REUSE0_USE_3RED, BGJ3_EPI8_USE_FARAWAY_CENTER, 0, 1>(bucket_epi8_t<BGJ3_EPI8_REUSE0_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
template int Pool_epi8_t<4>::_pool_bucketing<BGJ3_EPI8_BUCKET0_BATCHSIZE, BGJ3_EPI8_REUSE0_USE_3RED, BGJ3_EPI8_USE_FARAWAY_CENTER, 0, 0>(bucket_epi8_t<BGJ3_EPI8_REUSE0_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
#endif

template int Pool_epi8_t<4>::_sub_bucketing<BGJ2_EPI8_BUCKET1_BATCHSIZE, BGJ2_EPI8_USE_FARAWAY_CENTER, 1, 1, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::_sub_bucketing<BGJ2_EPI8_BUCKET1_BATCHSIZE, BGJ2_EPI8_USE_FARAWAY_CENTER, 1, 0, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::_sub_bucketing<BGJ3_EPI8_BUCKET1_BATCHSIZE, BGJ3_EPI8_USE_FARAWAY_CENTER, 0, 1, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::_sub_bucketing<BGJ3_EPI8_BUCKET1_BATCHSIZE, BGJ3_EPI8_USE_FARAWAY_CENTER, 0, 0, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
#if BGJ3_EPI8_BUCKET2_BATCHSIZE != BGJ2_EPI8_BUCKET2_BATCHSIZE
template int Pool_epi8_t<4>::_sub_bucketing<BGJ3_EPI8_BUCKET2_BATCHSIZE, BGJ3_EPI8_USE_FARAWAY_CENTER, 1, 1, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::_sub_bucketing<BGJ3_EPI8_BUCKET2_BATCHSIZE, BGJ3_EPI8_USE_FARAWAY_CENTER, 1, 0, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
#endif

#if BGJ_NEED_SEARCH_3RED
template int Pool_epi8_t<4>::_search_cred<1, 1>(bucket_epi8_t<1> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::_search_np<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 1, 1>(bucket_epi8_t<1> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::_search_pp<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 1, 1>(bucket_epi8_t<1> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::_search_nn<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 1, 1>(bucket_epi8_t<1> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
#endif

template int Pool_epi8_t<4>::_search_cred<0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::_search_np<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::_search_pp<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::_search_nn<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);

template uint64_t Pool_epi8_t<4>::_pool_insert<1>(sol_list_epi8_t **sol_list, long num_sol_list, int32_t goal_norm, int32_t goal_index, bgj_profile_data_t<4> *prof);
template uint64_t Pool_epi8_t<4>::_pool_insert<0>(sol_list_epi8_t **sol_list, long num_sol_list, int32_t goal_norm, int32_t goal_index, bgj_profile_data_t<4> *prof);
#endif

#if COMPILE_POOL_EPI8_160
template int Pool_epi8_t<5>::_pool_bucketing<BGJ1_EPI8_BUCKET_BATCHSIZE, BGJ1_EPI8_USE_3RED, BGJ1_EPI8_USE_FARAWAY_CENTER, 1, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
template int Pool_epi8_t<5>::_pool_bucketing<BGJ1_EPI8_BUCKET_BATCHSIZE, BGJ1_EPI8_USE_3RED, BGJ1_EPI8_USE_FARAWAY_CENTER, 1, 0>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
template int Pool_epi8_t<5>::_pool_bucketing<BGJ2_EPI8_BUCKET0_BATCHSIZE, BGJ2_EPI8_REUSE_USE_3RED, BGJ2_EPI8_USE_FARAWAY_CENTER, 0, 1>(bucket_epi8_t<BGJ2_EPI8_REUSE_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
template int Pool_epi8_t<5>::_pool_bucketing<BGJ2_EPI8_BUCKET0_BATCHSIZE, BGJ2_EPI8_REUSE_USE_3RED, BGJ2_EPI8_USE_FARAWAY_CENTER, 0, 0>(bucket_epi8_t<BGJ2_EPI8_REUSE_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
#if BGJ3_EPI8_BUCKET0_BATCHSIZE != BGJ2_EPI8_BUCKET0_BATCHSIZE
template int Pool_epi8_t<5>::_pool_bucketing<BGJ3_EPI8_BUCKET0_BATCHSIZE, BGJ3_EPI8_REUSE0_USE_3RED, BGJ3_EPI8_USE_FARAWAY_CENTER, 0, 1>(bucket_epi8_t<BGJ3_EPI8_REUSE0_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
template int Pool_epi8_t<5>::_pool_bucketing<BGJ3_EPI8_BUCKET0_BATCHSIZE, BGJ3_EPI8_REUSE0_USE_3RED, BGJ3_EPI8_USE_FARAWAY_CENTER, 0, 0>(bucket_epi8_t<BGJ3_EPI8_REUSE0_USE_3RED> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
#endif

template int Pool_epi8_t<5>::_sub_bucketing<BGJ2_EPI8_BUCKET1_BATCHSIZE, BGJ2_EPI8_USE_FARAWAY_CENTER, 1, 1, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_sub_bucketing<BGJ2_EPI8_BUCKET1_BATCHSIZE, BGJ2_EPI8_USE_FARAWAY_CENTER, 1, 0, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_sub_bucketing<BGJ3_EPI8_BUCKET1_BATCHSIZE, BGJ3_EPI8_USE_FARAWAY_CENTER, 0, 1, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_sub_bucketing<BGJ3_EPI8_BUCKET1_BATCHSIZE, BGJ3_EPI8_USE_FARAWAY_CENTER, 0, 0, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
#if BGJ3_EPI8_BUCKET2_BATCHSIZE != BGJ2_EPI8_BUCKET2_BATCHSIZE
template int Pool_epi8_t<5>::_sub_bucketing<BGJ3_EPI8_BUCKET2_BATCHSIZE, BGJ3_EPI8_USE_FARAWAY_CENTER, 1, 1, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_sub_bucketing<BGJ3_EPI8_BUCKET2_BATCHSIZE, BGJ3_EPI8_USE_FARAWAY_CENTER, 1, 0, 1>(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
#endif

#if BGJ_NEED_SEARCH_3RED
template int Pool_epi8_t<5>::_search_cred<1, 1>(bucket_epi8_t<1> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_search_np<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 1, 1>(bucket_epi8_t<1> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_search_pp<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 1, 1>(bucket_epi8_t<1> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_search_nn<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 1, 1>(bucket_epi8_t<1> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
#endif

template int Pool_epi8_t<5>::_search_cred<0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_search_np<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_search_pp<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_search_nn<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);

template uint64_t Pool_epi8_t<5>::_pool_insert<1>(sol_list_epi8_t **sol_list, long num_sol_list, int32_t goal_norm, int32_t goal_index, bgj_profile_data_t<5> *prof);
template uint64_t Pool_epi8_t<5>::_pool_insert<0>(sol_list_epi8_t **sol_list, long num_sol_list, int32_t goal_norm, int32_t goal_index, bgj_profile_data_t<5> *prof);
#endif