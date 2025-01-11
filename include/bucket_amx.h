#ifndef __BUCKET_AMX_H
#define __BUCKET_AMX_H

#include "pool_epi8.h"

struct bucket_amx_t {
    uint32_t center_ind;

    long num_pvec = 0;
    long num_nvec = 0;

    uint32_t *pvec = NULL;
    uint32_t *nvec = NULL;
    int32_t *pnorm = NULL;
    int32_t *nnorm = NULL;

    bucket_amx_t() {}
    bucket_amx_t (long size) { _alloc(size, 0); _alloc(size, 1); }
    ~bucket_amx_t() { _clear(); }

    inline void add_pvec(uint32_t _ind, int32_t _norm) {
        if (num_pvec == _psize) _alloc(_psize * 2 + 64, 1);
        pvec[num_pvec] = _ind;
        pnorm[num_pvec] = _norm;
        num_pvec++;
    }
    inline void add_nvec(uint32_t _ind, int32_t _norm) {
        if (num_nvec == _nsize) _alloc(_nsize * 2 + 64, 0);
        nvec[num_nvec] = _ind;
        nnorm[num_nvec] = _norm;
        num_nvec++;
    }

    void combine(bucket_amx_t **subbucket_list, long len);
    int remove_center(int max_unordered);

    int _clear();
    int _alloc(long size, bool p);

    long _psize = 0;
    long _nsize = 0;
};

struct sol_list_amx_t {
    uint32_t *a_list = NULL;
    uint32_t *s_list = NULL;
    long num_a = 0, num_s = 0;
    #if BOOST_AMX_SIEVE
    long num_a_insert = 0, num_s_insert = 0;
    double filter_time = 0.0;
    #endif

    sol_list_amx_t() {}
    ~sol_list_amx_t() { _clear(); }

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
    inline void init(long &status, long &status_ind) {
        status_ind = 0;
        if (num_a) { status = 0; return; }
        if (num_s) { status = 1; return; }
        status = 2;
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
            break;
        }
    }
    int _clear();
    int _alloc(long size, long type);

    long num_sol();

    long _asize = 0, _ssize = 0;
};

#endif