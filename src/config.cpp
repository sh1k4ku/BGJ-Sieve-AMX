#include "../include/config.h"

long check_avx512f() {
    #ifdef __AVX512F__
    return 1;
    #else
    return 0;
    #endif
}

long check_avx512vnni() {
    #ifdef __AVX512VNNI__
    return 1;
    #else
    return 0;
    #endif
}

long check_amx() {
    #ifdef __AMX_INT8__
    return 1;
    #else
    #warning "amx_int8 not found, disabling amx"
    return 0;
    #endif
}

long check_gso_blocksize() {
    return GSO_BLOCKSIZE;
}

long check_jumping_step() {
    return JUMPING_STEP;
}