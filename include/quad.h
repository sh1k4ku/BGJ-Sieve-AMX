#ifndef __QUAD_H
#define __QUAD_H

#include <NTL/quad_float.h>

//todo: an avx2 version.

/* a quad_float is represented by two doubles, hi and lo. We want
 * to vectorize the basic operations of quad_float. I will make sure
 * the result is correct on my machine, but it may have errors on 
 * other platforms (basically this is because the compiler want to 
 * use fma instead of mul and add instructions, which is actually 
 * more accuracy but is not what we want). So run "make test" to 
 * make sure it works as expect.    --2022.3.31
 * it seems that I don't know how to tell g++ to disable 
 * fma while enable other optimizations. I will use inline assemble 
 * code directly.   --2022.4.1
*/ 

//aligned quad_float vector
struct VEC_QP{
    double *hi;
    double *lo;
};
//aligned quad_float matrix
struct MAT_QP{
    double **hi;
    double **lo;
};


/* naive vec operations, n can be any integer, data need not to be aligned. */
NTL::quad_float dot(NTL::quad_float *src1, NTL::quad_float *src2, long n);          //return the dot product
void red(NTL::quad_float *dst, NTL::quad_float *src, NTL::quad_float q, long n);    //dst -= src*q
void copy(NTL::quad_float *dst, NTL::quad_float *src, long n);                      //dst = src
void mul(NTL::quad_float *dst, NTL::quad_float q, long n);                          //dst *= q
void sub(NTL::quad_float *dst, NTL::quad_float q, long n);                          //dst -= q


/* aligned vec operations, n should be divided by 8. */
NTL::quad_float dot(double *src1_hi, double *src1_lo, double *src2_hi, double *src2_lo, long n);        //return the dot product
void red(double *dst_hi, double *dst_lo, double *src_hi, double *src_lo, NTL::quad_float q, long n);    //dst -= src*q
void copy(double *dst_hi, double *dst_lo, double *src_hi, double *src_lo, long n);                      //dst = src
void mul(double *dst_hi, double *dst_lo, NTL::quad_float q, long n);                                    //dst *= q
void sub(double *dst_hi, double *dst_lo, NTL::quad_float q, long n);                                    //dst -= q


/* operations on VEC_QP */
NTL::quad_float dot(VEC_QP src1, VEC_QP src2, long n);                              //return the dot product
void red(VEC_QP dst, VEC_QP src, NTL::quad_float q, long n);                        //dst -= src*q
void copy(VEC_QP dst, VEC_QP src, long n);                                          //dst = src
void mul(VEC_QP dst, NTL::quad_float q, long n);                                    //dst *= q
void sub(VEC_QP dst, NTL::quad_float q, long n);                                    //dst -= q


#endif