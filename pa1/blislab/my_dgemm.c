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
 * bl_dgemm.c
 *
 *
 * Purpose:
 * this is the main file of blislab dgemm.
 *
 * Todo:
 *
 *
 * Modification:
 *      bryan chin - ucsd
 *      changed to row-major order  
 *      handle arbitrary  size C
 * */

#include <stdio.h>

#include "bl_dgemm_kernel.h"
#include "bl_dgemm.h"
const char* dgemm_desc = "my blislab ";


/* 
 * pack one subpanel of A
 *
 * pack like this 
 * if A is row major order
 *
 *     a c e g
 *     b d f h
 *     i k m o
 *     j l n p
 *     q r s t
 *     
 * then pack into a sub panel
 * each letter represents sequantial
 * addresses in the packed result
 * (e.g. a, b, c, d are sequential
 * addresses). 
 * - down each column
 * - then next column in sub panel
 * - then next sub panel down (on subseqent call)
 
 *     a c e g  < each call packs one
 *     b d f h  < subpanel
 *     -------
 *     i k m o
 *     j l n p
 *     -------
 *     q r s t
 *     0 0 0 0
 */
static inline
void packA_mcxkc_d(
        int    m,  // mr
        int    k,  // kc
        int    pad_row,
        int    pad_col,
        double * restrict XA,  // start of panel A
        int    ldXA,  // column num of A
        double * restrict packA  // start of pack A
        )
{
  int icol, irow;
  for (icol = 0; icol < k; ++icol) {
    for (irow = 0; irow < m; ++irow) {
      // Here we pack A[irow, icol] into packA sequentially
      *(packA + icol * (m + pad_row) + irow) = *(XA + irow * ldXA + icol);
    }
  }

  // Do Padding
  // TODO: Are there any better ways
  for (icol = k; icol < k + pad_col; ++icol) {
    for (irow = 0; irow < m; ++irow) {
      *(packA + icol * (m + pad_row) + irow) = 0;
    }
  }

  for (icol = 0; icol < k + pad_col; ++icol) {
    for (irow = m; irow < m + pad_row; ++irow) {
      *(packA + icol * (m + pad_row) + irow) = 0;
    }
  }
}



/*
 * --------------------------------------------------------------------------
 */

/* 
 * pack one subpanel of B
 * 
 * pack like this 
 * if B is 
 *
 * row major order matrix
 *     a b c j k l s t
 *     d e f m n o u v
 *     g h i p q r w x
 *
 * each letter represents sequantial
 * addresses in the packed result
 * (e.g. a, b, c, d are sequential
 * addresses). 
 *
 * Then pack 
 *   - across each row in the subpanel
 *   - then next row in each subpanel
 *   - then next subpanel (on subsequent call)
 *
 *     a b c |  j k l |  s t 0
 *     d e f |  m n o |  u v 0
 *     g h i |  p q r |  w x 0
 *
 *     ^^^^^
 *     each call packs one subpanel
 */
static inline
void packB_kcxnc_d(
        int    n,  // nr
        int    k,  // kc
        int    pad_row,
        int    pad_col,
        double * restrict XB,  // start of panel B
        int    ldXB,  // column num of B
        double * restrict packB  // start of pack B
        )
{
  int icol, irow;
  for (irow = 0; irow < k; ++irow) {
    for (icol = 0; icol < n; ++icol) {
      // Here we pack B[irow, icol] into packB sequentially
      *(packB + irow * (n + pad_col) + icol) = *(XB + irow * ldXB + icol);
    }
  }

  // Do Padding
  // TODO: Are there any better ways
  for (irow = k; irow < k + pad_row; ++irow) {
    for (icol = 0; icol < n; ++icol) {
      *(packB + irow * (n + pad_col) + icol) = 0;
    }
  }

  for (irow = 0; irow < k + pad_row; ++irow) {
    for (icol = n; icol < n + pad_col; ++icol) {
      *(packB + irow * (n + pad_col) + icol) = 0;
    }
  }
}

/*
 * --------------------------------------------------------------------------
 */

static
inline
void bl_macro_kernel(
        int    m,  // MC
        int    n,  // NC
        int    k,  // KC
        const double * restrict packA,
        const double * restrict packB,
        double * restrict C,
        int    ldc
        )
{
    int    i, j;
    for ( i = 0; i < m; i += DGEMM_MR ) {                      // 2-th loop around micro-kernel
      for ( j = 0; j < n; j += DGEMM_NR ) {                    // 1-th loop around micro-kernel
	( *bl_sve_micro_kernel ) (
			      DGEMM_KC,
			      DGEMM_MR,
			      DGEMM_NR,
			      &packA[i * k],          // assumes sq matrix, otherwise use lda
			      &packB[j * k],                // 

			      // what you should use after real packing routine implmemented
			      //			      &packA[ i * k ],
			      //			      &packB[ j * k ],
			      &C[ i * ldc + j ],
			      (unsigned long long) ldc
			      );
      }                                                        // 1-th loop around micro-kernel
    }                                                          // 2-th loop around micro-kernel
}

void bl_dgemm(
        int    m,
        int    n,
        int    k,
        double * restrict XA,
        int    lda,
        double * restrict XB,
        int    ldb,
        double * restrict C,       
        int    ldc       
        )
{
    int    ic, ib, jc, jb, pc, pb;
    int    pad_k;
    double * restrict packA, * restrict packB;

    // Allocate packing buffers
    // 
    // FIXME undef NOPACK when you implement packing
    //
    packA  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_MC/DGEMM_MR + 1 )* DGEMM_MR, sizeof(double) );
    packB  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_NC/DGEMM_NR + 1 )* DGEMM_NR, sizeof(double) );

    for ( ic = 0; ic < m; ic += DGEMM_MC ) {              // 5-th loop around micro-kernel
        ib = min( m - ic, DGEMM_MC );
        for ( pc = 0; pc < k; pc += DGEMM_KC ) {          // 4-th loop around micro-kernel
            pb = min( k - pc, DGEMM_KC );
            pad_k = DGEMM_KC - pb;
            int    i, j, ibi, jbj;

            for ( i = 0; i < ib; i += DGEMM_MR ) {
                ibi = ib - i;
                packA_mcxkc_d(
                    min( ibi, DGEMM_MR ), /* m */
                    pb,                      /* k */
                    ~((DGEMM_MR - ibi) >> 31) & (DGEMM_MR - ibi), // max(0, DGEMM_MR - ibi)
                    pad_k,
                    &XA[ pc + lda*(ic + i)], /* XA - start of micropanel in A */
                    k,                       /* ldXA */
                    &packA[ i * DGEMM_KC ] /* packA */
                );
            }

            for ( jc = 0; jc < n; jc += DGEMM_NC ) {        // 3-rd loop around micro-kernel
                jb = min( n - jc, DGEMM_NC );

                for ( j = 0; j < jb; j += DGEMM_NR ) {
                    jbj = jb - j;
                    packB_kcxnc_d(
                        min( jbj, DGEMM_NR ) /* n */,
                        pb                      /* k */,
                        pad_k,
                        ~((DGEMM_NR - jbj) >> 31) & (DGEMM_NR - jbj), // max(0, DGEMM_NR - jbj)
                        &XB[ ldb * pc + jc + j]     /* XB - starting row and column for this panel */,
                        n, // should be ldXB instead /* ldXB */
                        &packB[ j * DGEMM_KC ]        /* packB */
                    );
                }

                bl_macro_kernel(
                    ib,  // MC
                    jb,  // NC
                    DGEMM_KC,  // KC
                    packA,
                    packB,
                    &C[ ic * ldc + jc ], 
                    ldc
                );
            }// End 3.rd loop around micro-kernel
        }// End 4.th loop around micro-kernel
    }// End 5.th loop around micro-kernel

    free( packA );
    free( packB );
}

void square_dgemm(int lda, double * restrict A, double * restrict B, double * restrict C){
  bl_dgemm(lda, lda, lda, A, lda, B, lda, C,  lda);
}