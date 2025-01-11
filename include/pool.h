#ifndef __POOL_H
#define __POOL_H

#include <omp.h>
#include <fstream>
#include "lattice.h"
#include "utils.h"
#include "vec.h"
#include "UidHashTable.h"
#include "sampler.h"
#include "../dep/g6k/parallel_algorithms.hpp"
#include "../dep/g6k/thread_pool.hpp"

#define cvec_size 6
#define THREE_SIEVE_GET_STUCKED 7923

struct three_Sieve_params{
    double alpha = 0.3;
    double one_epoch_ratio = 0.025;
    double saturation_ratio = 0.43;
    double saturation_radius = 4.0/3.0;
    double improve_ratio = 0.65;
    double resort_ratio = 0.85;
};

struct fc_Sieve_params{
    bool show_details = true;
    //bucketing params
    long center_dim = 11;        //dimension of the center
    long search_radius = 1;     //we only search subbuckets with distance less than it
    float center_alpha = 0.475;   //the alpha to get into the center
    long batch_size = 24;       //number of buckets to search each epoch, can be divided by num_threads

    //simhash values to optimize
    long XPC_FCS_THRESHOLD = 96;

    //general sieving params    
    double improve_ratio = 0.65;
    double one_epoch_ratio = 0.025;
    double resort_ratio = 0.85;
    double saturation_radius = 4./3.;
    double saturation_ratio = 0.43;
};

class Pool {
    public:
        //basis
            Lattice_QP *basis;
            float **b_local = NULL;

        //Sieving Status
            long MSD;       //maximal sieving dimension
            long CSD;       //current sieving dimension
            long index_l;   //current sieving context = [index_l, index_r]
            long index_r;
            float gh2;      //gh^2 of L_{[index_l, index_r]}
            long vec_size;  
            long int_bias;
            long vec_length;

        //Simhash and uid
            uint32_t* compress_pos = NULL;
            UidHashTable *uid = NULL;

        //pool
            long max_pool_size = 0;
            long num_vec = 0;
            float *vec = NULL;
            long *cvec = NULL;

        //construction and distructions
            Pool();
            Pool(Lattice_QP *L);
            ~Pool();
            void clear_all();
            void clear_pool();

        //setup
            int set_num_threads(long n);
            int set_MSD(long msd);
            int set_max_pool_size(long N);
            int set_basis(Lattice_QP *L);
            int set_sieving_context(long l, long r);
            int compute_gh2();

        //pool operations
            //do gaussian sampling to collect N vectors in the pool
            int gaussian_sampling(long N);
            //shrink the pool size to N
            int shrink(long N);
            //extend_left
            int extend_left();
            //shrink left
            int shrink_left();
            //sort the cvec list
            int sort_cvec();
            //check whether sieve is over
            int sieve_is_over(double saturation_radius, double saturation_ratio, bool show_details = false);
            //do insertion
            int insert(long index, double delta);
            //show the minimal lift to index
            int show_min_lift(long index);
            //do LLL in the last n vector, maintain the pool.
            int tail_LLL(double delta, long n);
            //store the pool to file
            int store(const char *file_name);
            //load the pool from file
            int load(const char *file_name);
            //store the vectors in the pool
            int store_vec(const char *file_name);
            //return the pot of the lattice
            double pot();
            //check uid and simhash
            bool check_pool_status();
            //check lose of dimension
            void check_dim_lose();

        //Sieving
            //3-sieve, if sieve is not over after maxcc searches, THREE_SIEVE_GET_STUCK will be returned.
            int three_Sieve(three_Sieve_params params, int show_details, long maxcc);
            // a more parallel version for 3-sieve
            int three_Sieve_parallel(three_Sieve_params params, int show_details, long maxcc);
            int fc_Sieve(fc_Sieve_params params);
            int bgj1_Sieve(long log_level = 0, long lps_auto_adj = 1, long num_empty = 0);
            int bgj2_Sieve(long log_level = 0, long lps_auto_adj = 1, long num_empty = 0);
            int bgj3_Sieve(long log_level = 0, long lps_auto_adj = 1, long num_empty = 0);
            //left progressive_sieve on L_{[ind_l, ind_r]}
            int left_progressive_3sieve(long ind_l, long ind_r, long num_threads, int show_details);
            int left_progressive_bgj1sieve(long ind_l, long ind_r, long num_threads, long log_level);
            int left_progressive_bgj2sieve(long ind_l, long ind_r, long num_threads, long log_level);
    private:
        float *vec_store = NULL;
        long *cvec_store = NULL;
        long num_threads = 1;
        long sorted_index = 0;
        thread_pool::thread_pool threadpool;
        void Simhash_setup();
        void update_b_local();
        void gaussian_sampling(float *res, long *cres, DGS1d &R);
        inline void compute_vec(float *res);
        inline void compute_Simhash(float *res);
        inline void compute_uid(float *res);
        inline void compute_cvec(float *res, long *cres);
};

class coeff_buffer{
    public:
        coeff_buffer(){};
        coeff_buffer(long coeffsize, long maxsize);
        ~coeff_buffer();
        void buffer_setup(long coeffsize, long maxsize);

        __attribute__ ((aligned (128))) long size = 0;
        long coeff_size;
        long max_size;
        short *buffer;
        short *buffer_store = NULL;
};

struct size{
    __attribute__ ((aligned (128))) int a = 0;
};

inline void Pool::compute_vec(float *res){
    short *x = (short *)(&res[-int_bias]);
    set_zero(res, vec_length);
    for (long i = 0; i < CSD; i++){
        red(res, b_local[i], -x[CSD-1-i], CSD);
    }
    compute_uid(res);
    compute_Simhash(res);
    res[-1] = norm(res, vec_length);
}

/***\
*   I learned this simhash implementation from G6K, the code is similar
*
*   Copyright (C) 2018-2021 Team G6K
*
*   This file is part of G6K. G6K is free software:
*   you can redistribute it and/or modify it under the terms of the
*   GNU General Public License as published by the Free Software Foundation,
*   either version 2 of the License, or (at your option) any later version.
*
*   G6K is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with G6K. If not, see <http://www.gnu.org/licenses/>.
*
****/
inline void Pool::compute_Simhash(float *res){
    uint64_t c0, c1, c2, c3;
    float a[4];
    uint32_t *a0 = (uint32_t *)(&a[0]);
    uint32_t *a1 = (uint32_t *)(&a[1]);
    uint32_t *a2 = (uint32_t *)(&a[2]);
    uint32_t *a3 = (uint32_t *)(&a[3]);

    c0 = 0;
    c1 = 0;
    c2 = 0;
    c3 = 0;
    for (long i = 0; i < 64; i++){
        a[0]   = res[compress_pos[24*i+0]];
        a[0]  += res[compress_pos[24*i+1]];
        a[0]  += res[compress_pos[24*i+2]];
        a[0]  -= res[compress_pos[24*i+3]];
        a[0]  -= res[compress_pos[24*i+4]];
        a[0]  -= res[compress_pos[24*i+5]];

        a[1]   = res[compress_pos[24*i+6]];
        a[1]  += res[compress_pos[24*i+7]];
        a[1]  += res[compress_pos[24*i+8]];
        a[1]  -= res[compress_pos[24*i+9]];
        a[1]  -= res[compress_pos[24*i+10]];
        a[1]  -= res[compress_pos[24*i+11]];

        a[2]   = res[compress_pos[24*i+12]];
        a[2]  += res[compress_pos[24*i+13]];
        a[2]  += res[compress_pos[24*i+14]];
        a[2]  -= res[compress_pos[24*i+15]];
        a[2]  -= res[compress_pos[24*i+16]];
        a[2]  -= res[compress_pos[24*i+17]];

        a[3]   = res[compress_pos[24*i+18]];
        a[3]  += res[compress_pos[24*i+19]];
        a[3]  += res[compress_pos[24*i+20]];
        a[3]  -= res[compress_pos[24*i+21]];
        a[3]  -= res[compress_pos[24*i+22]];
        a[3]  -= res[compress_pos[24*i+23]];

        c0 |= (uint64_t)((a0[0] & 0x80000000) >> 31) << i;
        c1 |= (uint64_t)((a1[0] & 0x80000000) >> 31) << i;
        c2 |= (uint64_t)((a2[0] & 0x80000000) >> 31) << i;
        c3 |= (uint64_t)((a3[0] & 0x80000000) >> 31) << i;
    } 
    *((uint64_t *)(&res[-16])) = c0;
    *((uint64_t *)(&res[-14])) = c1;
    *((uint64_t *)(&res[-12])) = c2;
    *((uint64_t *)(&res[-10])) = c3;
}
inline void Pool::compute_uid(float *res){
    short *x = (short *)(&res[-int_bias]);
    uint64_t u = 0;
    for (long i = 0; i < CSD; i++){
        u += x[i] * uid->uid_coeffs[i];
    }
    *((uint64_t *)(&res[-4])) = u;
}
inline void Pool::compute_cvec(float *res, long *cres){
    *((uint64_t *)(&cres[0])) = *((uint64_t *)(&res[-16]));
    *((uint64_t *)(&cres[1])) = *((uint64_t *)(&res[-14]));
    *((uint64_t *)(&cres[2])) = *((uint64_t *)(&res[-12]));
    *((uint64_t *)(&cres[3])) = *((uint64_t *)(&res[-10]));
    *((float *)(&cres[4])) = res[-1];
    *((float **)(&cres[5])) = res;
}


#endif