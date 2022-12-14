#include "bl_config.h"
#include "bl_dgemm_kernel.h"

void bl_sve_dgemm_ukr16X4( int k,
		   int m,
		   int n,
        double * restrict m_a,
        double * restrict m_b,
        double * restrict m_c,
        unsigned long long ldc) {
  /**
   * Function Instructions
   * VLEN = 256b thus svfloat64_t stores 4 doubles
   * 
   * svwhilelt_b64_u64(i, j) - return predicate s.t. i + sth < j
   * svldl_f64(p, base) - return vector with active element i contains base[i] & others 0
   * svdup_f64(val) - return vector where all element i contains val
   * svmla(p, v1, v2, v3) - return vector with active element i = v1[i] + v2[i] * v3[i]
   * svst1_f64(p, base, data) - store active data[i] to base[i]
  */
  register svfloat64_t ax;
  register svfloat64_t bx;
  register svfloat64_t c0x, c1x, c2x, c3x, c4x, c5x, c6x, c7x, c8x, c9x, c10x, c11x, c12x, c13x, c14x, c15x;
  svbool_t npred = svwhilelt_b64_u64(0, 4);
  c0x = svld1_f64(npred, m_c);
  c1x = svld1_f64(npred, m_c + ldc);
  c2x = svld1_f64(npred, m_c + ldc * 2);
  c3x = svld1_f64(npred, m_c + ldc * 3);
  c4x = svld1_f64(npred, m_c + ldc * 4);
  c5x = svld1_f64(npred, m_c + ldc * 5);
  c6x = svld1_f64(npred, m_c + ldc * 6);
  c7x = svld1_f64(npred, m_c + ldc * 7);
  c8x = svld1_f64(npred, m_c + ldc * 8);
  c9x = svld1_f64(npred, m_c + ldc * 9);
  c10x = svld1_f64(npred, m_c + ldc * 10);
  c11x = svld1_f64(npred, m_c + ldc * 11);
  c12x = svld1_f64(npred, m_c + ldc * 12);
  c13x = svld1_f64(npred, m_c + ldc * 13);
  c14x = svld1_f64(npred, m_c + ldc * 14);
  c15x = svld1_f64(npred, m_c + ldc * 15);
  for (int p = 0; p < k; ++p) {
    // c0x
    register double aval = MATRIX_ACCESS( a, p, 0, m );
    ax = svdup_f64(aval);
    bx = svld1_f64(svptrue_b64(), m_b + p * n);
    c0x = svmla_f64_m(npred, c0x, bx, ax);
    // c1x
    aval = MATRIX_ACCESS( a, p, 1, m );
    ax = svdup_f64(aval);
    c1x = svmla_f64_m(npred, c1x, bx, ax);
    // c2x
    aval = MATRIX_ACCESS( a, p, 2, m );
    ax = svdup_f64(aval);
    c2x = svmla_f64_m(npred, c2x, bx, ax);
    // c3x
    aval = MATRIX_ACCESS( a, p, 3, m );
    ax = svdup_f64(aval);
    c3x = svmla_f64_m(npred, c3x, bx, ax);
    // c4x
    aval = MATRIX_ACCESS( a, p, 4, m );
    ax = svdup_f64(aval);
    c4x = svmla_f64_m(npred, c4x, bx, ax);
    // c5x
    aval = MATRIX_ACCESS( a, p, 5, m );
    ax = svdup_f64(aval);
    c5x = svmla_f64_m(npred, c5x, bx, ax);
    // c6x
    aval = MATRIX_ACCESS( a, p, 6, m );
    ax = svdup_f64(aval);
    c6x = svmla_f64_m(npred, c6x, bx, ax);
    // c7x
    aval = MATRIX_ACCESS( a, p, 7, m );
    ax = svdup_f64(aval);
    c7x = svmla_f64_m(npred, c7x, bx, ax);
    // c8x
    aval = MATRIX_ACCESS( a, p, 8, m );
    ax = svdup_f64(aval);
    c8x = svmla_f64_m(npred, c8x, bx, ax);
    // c9x
    aval = MATRIX_ACCESS( a, p, 9, m );
    ax = svdup_f64(aval);
    c9x = svmla_f64_m(npred, c9x, bx, ax);
    // c10x
    aval = MATRIX_ACCESS( a, p, 10, m );
    ax = svdup_f64(aval);
    c10x = svmla_f64_m(npred, c10x, bx, ax);
    // c11x
    aval = MATRIX_ACCESS( a, p, 11, m );
    ax = svdup_f64(aval);
    c11x = svmla_f64_m(npred, c11x, bx, ax);
    // c12x
    aval = MATRIX_ACCESS( a, p, 12, m );
    ax = svdup_f64(aval);
    c12x = svmla_f64_m(npred, c12x, bx, ax);
    // c13x
    aval = MATRIX_ACCESS( a, p, 13, m );
    ax = svdup_f64(aval);
    c13x = svmla_f64_m(npred, c13x, bx, ax);
    // c14x
    aval = MATRIX_ACCESS( a, p, 14, m );
    ax = svdup_f64(aval);
    c14x = svmla_f64_m(npred, c14x, bx, ax);
    // c15x
    aval = MATRIX_ACCESS( a, p, 15, m );
    ax = svdup_f64(aval);
    c15x = svmla_f64_m(npred, c15x, bx, ax);
  }
  svst1_f64(npred, m_c, c0x);
  svst1_f64(npred, m_c + ldc, c1x);
  svst1_f64(npred, m_c + ldc * 2, c2x);
  svst1_f64(npred, m_c + ldc * 3, c3x);
  svst1_f64(npred, m_c + ldc * 4, c4x);
  svst1_f64(npred, m_c + ldc * 5, c5x);
  svst1_f64(npred, m_c + ldc * 6, c6x);
  svst1_f64(npred, m_c + ldc * 7, c7x);
  svst1_f64(npred, m_c + ldc * 8, c8x);
  svst1_f64(npred, m_c + ldc * 9, c9x);
  svst1_f64(npred, m_c + ldc * 10, c10x);
  svst1_f64(npred, m_c + ldc * 11, c11x);
  svst1_f64(npred, m_c + ldc * 12, c12x);
  svst1_f64(npred, m_c + ldc * 13, c13x);
  svst1_f64(npred, m_c + ldc * 14, c14x);
  svst1_f64(npred, m_c + ldc * 15, c15x);
}

// cse260
// you can put your optimized kernels here
// - put the function prototypes in bl_dgemm_kernel.h
// - define BL_MICRO_KERNEL appropriately in bl_config.h
//
