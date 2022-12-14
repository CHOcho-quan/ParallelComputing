/*
 * --------------------------------------------------------------------------
 * BLISLAB 
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * bl_config.h
 *
 *
 * Purpose:
 * this header file contains configuration parameters.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */

#ifndef BLISLAB_CONFIG_H
#define BLISLAB_CONFIG_H

// Allow C++ users to include this header file in their source code. However,
// we make the extern "C" conditional on whether we're using a C++ compiler,
// since regular C compilers don't understand the extern "C" construct.
#ifdef __cplusplus
extern "C" {
#endif

#define GEMM_SIMD_ALIGN_SIZE 32

#if 1
/**
 * Neo Verse V1 Parameter
 * L1 DCache - 64KB = 64000 / 8 = 8000 doubles ~= 89 x 89 matrix
 * L2 Cache - 512KB / 1MB = 64000 doubles / 128000 doubles ~= 252 x 252 matrix
*/
/*
#define DGEMM_KC 32
#define DGEMM_MC 256
#define DGEMM_NC 1024
#define DGEMM_MR 16
#define DGEMM_NR 4
*/
#ifndef DGEMM_KC
    #define DGEMM_KC 68
#endif
#ifndef DGEMM_MC
    #define DGEMM_MC 1088
#endif
#ifndef DGEMM_NC
    #define DGEMM_NC 2176
#endif
#ifndef DGEMM_MR
    #define DGEMM_MR 16
#endif
#ifndef DGEMM_NR
    #define DGEMM_NR 4
#endif
#endif

#define BL_MICRO_KERNEL bl_dgemm_ukr
#define SVE_KERNEL(m, k, n) bl_sve_dgemm_ukr_##m##x##k##x##n

// End extern "C" construct block.
#ifdef __cplusplus
}
#endif

#endif
