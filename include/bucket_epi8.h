#ifndef __BUCKET_EPI8_H
#define __BUCKET_EPI8_H

#include "pool_epi8.h"

template <bool record_dp>
struct bucket_epi8_t {
    uint32_t center_ind;
    int32_t center_norm;
    uint64_t center_u;

    long num_pvec = 0;
    long num_nvec = 0;
    uint32_t *pvec = NULL;
    uint32_t *nvec = NULL;
    int32_t *pnorm = NULL;
    int32_t *nnorm = NULL;
    int32_t *psum = NULL;
    int32_t *nsum = NULL;
    int32_t *pdot = NULL;
    int32_t *ndot = NULL;

    bucket_epi8_t() {}
    bucket_epi8_t(long size) { _alloc(size, 0); _alloc(size, 1); }
    ~bucket_epi8_t() { _clear(); }

    void set_center(uint32_t ind, int32_t norm, uint64_t u) {
        center_ind = ind;
        center_norm = norm;
        center_u = u;
    }
    void combine(bucket_epi8_t<record_dp> **subbucket_list, long len);
    // record_dp = true only 
    void add_vec(uint32_t _ind, int32_t _norm, int32_t _sum, int32_t _dot) {
        if (_dot > 0) {
            if (num_pvec == _psize) _alloc(_psize * 2 + 64, 1);
            pvec[num_pvec] = _ind;
            pnorm[num_pvec] = _norm;
            psum[num_pvec] = _sum;
            pdot[num_pvec] = _dot;
            num_pvec++;
        } else {
            if (num_nvec == _nsize) _alloc(_nsize * 2 + 64, 0);
            nvec[num_nvec] = _ind;
            nnorm[num_nvec] = _norm;
            nsum[num_nvec] = _sum;
            ndot[num_nvec] = _dot;
            num_nvec++;
        }
    }
    // record_dp = false
    void add_pvec(uint32_t _ind, int32_t _norm, int32_t _sum) {
        if (num_pvec == _psize) _alloc(_psize * 2 + 64, 1);
        pvec[num_pvec] = _ind;
        pnorm[num_pvec] = _norm;
        psum[num_pvec] = _sum;
        num_pvec++;
    }
    void add_nvec(uint32_t _ind, int32_t _norm, int32_t _sum) {
        if (num_nvec == _nsize) _alloc(_nsize * 2 + 64, 0);
        nvec[num_nvec] = _ind;
        nnorm[num_nvec] = _norm;
        nsum[num_nvec] = _sum;
        num_nvec++;
    }
    int remove_center(int max_unordered);

    int _clear();
    int _alloc(long size, bool p);

    long _psize = 0;
    long _nsize = 0;
};

struct sol_list_epi8_t {
    uint32_t *a_list = NULL;
    uint32_t *s_list = NULL;
    uint32_t *aa_list = NULL;
    uint32_t *sa_list = NULL;
    uint32_t *ss_list = NULL;
    long num_a = 0, num_s = 0;
    long num_aa = 0, num_sa = 0, num_ss = 0;

    sol_list_epi8_t() {}
    ~sol_list_epi8_t() { _clear(); }

    inline void add_sol_a(uint32_t src1, uint32_t src2){
        if (num_a == _asize) _alloc(2 * _asize + 64, 0);
        a_list[num_a * 2] = src1;
        a_list[num_a * 2 + 1] = src2;
        num_a++;
    }
    inline void add_sol_s(uint32_t src1, uint32_t src2){
        if (num_s == _ssize) _alloc(2 * _ssize + 64, 1);
        s_list[num_s * 2] =  src1;
        s_list[num_s * 2 + 1] = src2;
        num_s++;
    }
    inline void add_sol_aa(uint32_t srcc, uint32_t src1, uint32_t src2){
        if (num_aa == _aasize) _alloc(2 * _aasize + 64, 2);
        aa_list[num_aa * 3] = srcc;
        aa_list[num_aa * 3 + 1] = src1;
        aa_list[num_aa * 3 + 2] = src2;
        num_aa++;
    }
    inline void add_sol_sa(uint32_t srcc, uint32_t src1, uint32_t src2){
        if (num_sa == _sasize) _alloc(2 * _sasize + 64, 3);
        sa_list[num_sa * 3] = srcc;
        sa_list[num_sa * 3 + 1] = src1;
        sa_list[num_sa * 3 + 2] = src2;
        num_sa++;
    }
    inline void add_sol_ss(uint32_t srcc, uint32_t src1, uint32_t src2){
        if (num_ss == _sssize) _alloc(2 * _sssize + 64, 4);
        ss_list[num_ss * 3] = srcc;
        ss_list[num_ss * 3 + 1] = src1;
        ss_list[num_ss * 3 + 2] = src2;
        num_ss++;
    }
    inline void init(long &status, long &status_ind) {
        status_ind = 0;
        if (num_a) { status = 0; return; }
        if (num_s) { status = 1; return; }
        if (num_aa) { status = 2; return; }
        if (num_sa) { status = 3; return; }
        if (num_ss) { status = 4; return; }
        status = 5;
        return;
    }
    inline void next(long &status, long &status_ind) {
        status_ind++;
        while (1) {
            if (status == 0 && status_ind >= num_a) {
                status++;
                status_ind = 0;
                continue;
            } 
            if (status == 1 && status_ind >= num_s) {
                status++;
                status_ind = 0;
                continue;
            } 
            if (status == 2 && status_ind >= num_aa) {
                status++;
                status_ind = 0;
                continue;
            } 
            if (status == 3 && status_ind >= num_sa) {
                status++;
                status_ind = 0;
                continue;
            } 
            if (status == 4 && status_ind >= num_ss) {
                status++;
                status_ind = 0;
                continue;
            } 
            break;
        }
    }
    int _clear();
    int _alloc(long size, long type);

    long num_sol();

    long _asize = 0, _ssize = 0;
    long _aasize = 0, _sasize = 0, _sssize = 0;
};


#endif