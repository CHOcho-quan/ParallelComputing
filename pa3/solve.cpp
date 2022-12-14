/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */

#include <assert.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>
#ifdef _MPI_
#include <mpi.h>
#endif
using namespace std;

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);
double *alloc1D(int m, int n);


extern control_block cb;
extern std::vector<int> anx, any;
#ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
__attribute__((optimize("no-tree-vectorize")))
#endif 

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq){
    double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

void printMat2(const char mesg[], double *E, int m, int n){
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m>34)
      return;
#endif
    printf("%s\n",mesg);
    for (i=0; i < (m+2)*(n+2); i++){
       int rowIndex = i / (n+2);
       int colIndex = i % (n+2);
       if ((colIndex>0) && (colIndex<n+1))
          if ((rowIndex > 0) && (rowIndex < m+1))
            printf("%6.3f ", E[i]);
       if (colIndex == n+1)
	    printf("\n");
    }
}

#ifndef _MPI_

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){

 // Simulated time is different from the integer timestep number
 double t = 0.0;

 double *E = *_E, *E_prev = *_E_prev;
 double *R_tmp = R;
 double *E_tmp = *_E;
 double *E_prev_tmp = *_E_prev;
 double mx, sumSq;
 int niter;
 int m = cb.m, n=cb.n;
 int innerBlockRowStartIndex = (n+2)+1;
 int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);


 // We continue to sweep over the mesh until the simulation has reached
 // the desired number of iterations
  for (niter = 0; niter < cb.niters; niter++){
  
      if  (cb.debug && (niter==0)){
	  stats(E_prev,m,n,&mx,&sumSq);
          double l2norm = L2Norm(sumSq);
	  repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
	  if (cb.plot_freq)
	      plotter->updatePlot(E,  -1, m+1, n+1);
      }

   /* 
    * Copy data from boundary of the computational box to the
    * padding region, set up for differencing computational box's boundary
    *
    * These are physical boundary conditions, and are not to be confused
    * with ghost cells that we would use in an MPI implementation
    *
    * The reason why we copy boundary conditions is to avoid
    * computing single sided differences at the boundaries
    * which increase the running time of solve()
    *
    */
    
    // 4 FOR LOOPS set up the padding needed for the boundary conditions
    int i,j;

    // Fills in the TOP Ghost Cells
    for (i = 0; i < (n+2); i++) {
        E_prev[i] = E_prev[i + (n+2)*2];
    }

    // Fills in the RIGHT Ghost Cells
    for (i = (n+1); i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i-2];
    }

    // Fills in the LEFT Ghost Cells
    for (i = 0; i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i+2];
    }	

    // Fills in the BOTTOM Ghost Cells
    for (i = ((m+2)*(n+2)-(n+2)); i < (m+2)*(n+2); i++) {
        E_prev[i] = E_prev[i - (n+2)*2];
    }

//////////////////////////////////////////////////////////////////////////////

#define FUSED 1

#ifdef FUSED
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
	E_prev_tmp = E_prev + j;
        R_tmp = R + j;
	for(i = 0; i < n; i++) {
	    E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#else
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for(i = 0; i < n; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            }
    }

    /* 
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
        R_tmp = R + j;
	E_prev_tmp = E_prev + j;
        for(i = 0; i < n; i++) {
	  E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
	  R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#endif
     /////////////////////////////////////////////////////////////////////////////////

   if (cb.stats_freq){
     if ( !(niter % cb.stats_freq)){
        stats(E,m,n,&mx,&sumSq);
        double l2norm = L2Norm(sumSq);
        repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
    }
   }

   if (cb.plot_freq){
          if (!(niter % cb.plot_freq)){
	    plotter->updatePlot(E,  niter, m, n);
        }
    }

   // Swap current and previous meshes
   double *tmp = E; E = E_prev; E_prev = tmp;

 } //end of 'niter' loop at the beginning

  //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters

  stats(E_prev,m,n,&Linf,&sumSq);
  L2 = L2Norm(sumSq);

  // Swap pointers so we can re-use the arrays
  *_E = E;
  *_E_prev = E_prev;
}

#else // _MPI_
#define TAG_TOP 0
#define TAG_BOT 1
#define TAG_LFT 2
#define TAG_RGH 3

void solve(double ** _E, double ** _E_prev, double * R, double alpha, double dt, Plotter * plotter, double &L2, double &Linf){
    // Simulated time is different from the integer timestep number
    double t = 0.0;

    double *E = *_E, *E_prev = *_E_prev;
    double *R_tmp = R;
    double *E_tmp = *_E;
    double *E_prev_tmp = *_E_prev;
    double mx, sumSq;
    double pLinf, pSumSq;

    int niter, nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    register int rx = myrank / cb.py, ry = myrank % cb.py;
    int m = anx[rx], n = any[ry];
    // if (rx + 1 == cb.px && cb.m % cb.px) m = cb.m - (m * rx);
    // if (ry + 1 == cb.py && cb.n % cb.py) n = cb.n - (n * ry);

    // printf("===========================RANK IN SOLVE %d===========================\n===========================NX: %d, NY: %d, rx: %d, ry: %d===========================\n", myrank, m, n, rx, ry);
    // int innerBlockRowStartIndex = (n+2)*2+2;
    // int innerBlockRowEndIndex = m*(n+2)+n;
    register int innerBlockRowStartIndex = (n+2)+1;
    register int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);
    register int i, j;

    double *E_prev_tmp_col_left_i = alloc1D(m, 1);
    double *E_prev_tmp_col_left_o = alloc1D(m, 1);
    double *E_prev_tmp_col_right_i = alloc1D(m,1);
    double *E_prev_tmp_col_right_o = alloc1D(m,1);

    // Strided vector
    // static MPI_Datatype MPI_COL_VEC;
    // MPI_Type_vector(m, 1, (n+2), MPI_DOUBLE, &MPI_COL_VEC);
    // MPI_Type_commit(&MPI_COL_VEC);
 // We continue to sweep over the mesh until the simulation has reached
 // the desired number of iterations
    for (niter = 0; niter < cb.niters; niter++){
        // if (myrank == 0 && niter % 100 == 0) printf("Cur Iter %d\n", niter);
        if  (cb.debug && (niter==0)){
        stats(E_prev,m,n,&mx,&sumSq);
            double l2norm = L2Norm(sumSq);
        repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
        if (cb.plot_freq)
            plotter->updatePlot(E,  -1, m+1, n+1);
        }

        if (!cb.noComm) {
            // Async get ghost cells
            MPI_Request rec_requests[4], send_requests[4];
            MPI_Status recv_status[4];
            // printf("We're about to send MPI Stuff\n");
            // TOP Ghost cells
            if (rx == 0) { // top blocks - no need for send / receive
                #pragma GCC ivdep
                for (i = 0; i < (n+2); i++) E_prev[i] = E_prev[i + (n+2)*2];
            } else {
                MPI_Isend(&E_prev[1 + (n + 2) * 1], n, MPI_DOUBLE, myrank - cb.py, TAG_BOT, MPI_COMM_WORLD, &send_requests[0]);
                MPI_Irecv(&E_prev[1], n, MPI_DOUBLE, myrank - cb.py, TAG_TOP, MPI_COMM_WORLD, &rec_requests[0]);
            }
            // RIGHT Ghost cells
            if (ry + 1 == cb.py) {
                for (i = (n+1); i < (m+2)*(n+2); i+=(n+2)) E_prev[i] = E_prev[i-2];
            } else {
                MPI_Irecv(E_prev_tmp_col_right_i, m, MPI_DOUBLE, myrank + 1, TAG_RGH, MPI_COMM_WORLD, &rec_requests[1]);
                #pragma GCC ivdep
                for (i = 1; i < m+1; ++i) E_prev_tmp_col_right_o[i-1] = E_prev[((n + 2) * i + n)];
                MPI_Isend(E_prev_tmp_col_right_o, m, MPI_DOUBLE, myrank + 1, TAG_LFT, MPI_COMM_WORLD, &send_requests[1]);
                // MPI_Isend(&E_prev[n + (n + 2) * 1], 1, MPI_COL_VEC, myrank + 1, TAG_LFT, MPI_COMM_WORLD, &send_requests[1]);
                // MPI_Irecv(&E_prev[n + 1 + (n + 2) * 1], 1, MPI_COL_VEC, myrank + 1, TAG_RGH, MPI_COMM_WORLD, &rec_requests[1]);
            }
            // LEFT Ghost cells
            if (ry == 0) {
                for (i = 0; i < (m+2)*(n+2); i+=(n+2)) E_prev[i] = E_prev[i+2];
            } else {
                MPI_Irecv(E_prev_tmp_col_left_i, m, MPI_DOUBLE, myrank - 1, TAG_LFT, MPI_COMM_WORLD, &rec_requests[2]);
                #pragma GCC ivdep
                for (i = 1; i < m+1; ++i) E_prev_tmp_col_left_o[i-1] = E_prev[((n + 2) * i + 1)];
                MPI_Isend(E_prev_tmp_col_left_o, m, MPI_DOUBLE, myrank - 1, TAG_RGH, MPI_COMM_WORLD, &send_requests[2]);
                // MPI_Isend(&E_prev[1 + (n + 2) * 1], 1, MPI_COL_VEC, myrank - 1, TAG_RGH, MPI_COMM_WORLD, &send_requests[2]);
                // MPI_Irecv(&E_prev[(n + 2) * 1], 1, MPI_COL_VEC, myrank - 1, TAG_LFT, MPI_COMM_WORLD, &rec_requests[2]);
            }
            // BOTTOM Ghost cells
            if (rx + 1 == cb.px) { // top blocks - no need for send / receive
                #pragma GCC ivdep
                for (i = ((m+2)*(n+2)-(n+2)); i < (m+2)*(n+2); i++) E_prev[i] = E_prev[i - (n+2)*2];
            } else {
                MPI_Isend(&E_prev[1 + (n + 2) * m], n, MPI_DOUBLE, myrank + cb.py, TAG_TOP, MPI_COMM_WORLD, &send_requests[3]);
                MPI_Irecv(&E_prev[1 + (n + 2) * (m + 1)], n, MPI_DOUBLE, myrank + cb.py, TAG_BOT, MPI_COMM_WORLD, &rec_requests[3]);
            }
            // printf("Safe after send & recv async\n");

            if (ry+1 != cb.py) {
                MPI_Wait(&rec_requests[1], &recv_status[1]);
                #pragma GCC ivdep
                for (i = 1; i < m+1; ++i) E_prev[((n + 2) * i + n + 1)] = E_prev_tmp_col_right_i[i-1];
                // MPI_Wait(&send_requests[1], &send_status[1]);
            }

            if (ry != 0) {
                MPI_Wait(&rec_requests[2], &recv_status[2]);
                #pragma GCC ivdep
                for (i = 1; i < m+1; ++i) E_prev[((n + 2) * i)] = E_prev_tmp_col_left_i[i-1];
            }
            if (rx != 0) {
                MPI_Wait(&rec_requests[0], &recv_status[0]);
                // MPI_Wait(&send_requests[0], &send_status[0]);
            }
            if (rx+1 != cb.px) {
                MPI_Wait(&rec_requests[3], &recv_status[3]);
                // MPI_Wait(&send_requests[3], &send_status[3]);
            }
        }
//////////////////////////////////////////////////////////////////////////////
// DO Inner A-P Model when async sending & receiving
#define FUSED 1

#define SSE_VEC 1

#ifdef SSE_VEC
        __m128d mm_alph = _mm_set1_pd(alpha);
        __m128d mm_A = _mm_set1_pd(a);
        __m128d mm_b = _mm_set1_pd(b);
        __m128d mm_dt = _mm_set1_pd(dt);
        __m128d mm_kk = _mm_set1_pd(kk);
        __m128d mm_M1 = _mm_set1_pd(M1);
        __m128d mm_M2 = _mm_set1_pd(M2);
        __m128d mm_eps = _mm_set1_pd(epsilon);

        __m128d mm_four = _mm_set1_pd(4);
        __m128d mm_min1 = _mm_set1_pd(-1);
#endif

#ifdef FUSED
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
	    E_prev_tmp = E_prev + j;
        R_tmp = R + j;

#ifdef SSE_VEC
	    for(i = 0; i < n; i+=2) {
            __m128d E_I, E_T, E_B, E_L, E_R;
            __m128d E_IMinA, E_IMin1, E_IMulR;
            __m128d Tmp1, Tmp2, Tmp3, Tmp4, kkMulE_I, M1MulR, E_IAddM2, E_IMinB1;
            __m128d ETmp, ETmp_Tmp, RTmp;
            E_I = _mm_loadu_pd(&E_prev_tmp[i]);
            E_T = _mm_loadu_pd(&E_prev_tmp[i - (n + 2)]);
            E_B = _mm_loadu_pd(&E_prev_tmp[i + (n + 2)]);
            E_L = _mm_loadu_pd(&E_prev_tmp[i - 1]);
            E_R = _mm_loadu_pd(&E_prev_tmp[i + 1]);
            RTmp = _mm_loadu_pd(&R_tmp[i]);

	        // E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            Tmp1 = _mm_add_pd(E_T, E_B);
            Tmp2 = _mm_add_pd(E_L, E_R);
            Tmp3 = _mm_mul_pd(E_I, mm_four);
            Tmp4 = _mm_sub_pd(_mm_add_pd(Tmp1, Tmp2), Tmp3);
            ETmp_Tmp = _mm_add_pd(E_I, _mm_mul_pd(mm_alph, Tmp4));

            // E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            kkMulE_I = _mm_mul_pd(mm_kk, E_I);
            E_IMinA = _mm_sub_pd(E_I, mm_A);
            E_IMin1 = _mm_add_pd(E_I, mm_min1);
            E_IMulR = _mm_mul_pd(E_I, RTmp);
            Tmp1 = _mm_mul_pd(E_IMinA, E_IMin1);
            ETmp = _mm_sub_pd(ETmp_Tmp, _mm_mul_pd(mm_dt, _mm_add_pd(_mm_mul_pd(kkMulE_I, Tmp1), E_IMulR)));
            _mm_storeu_pd(&E_tmp[i], ETmp);

            // R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
            M1MulR = _mm_mul_pd(mm_M1, RTmp);
            E_IAddM2 = _mm_add_pd(E_I, mm_M2);
            E_IMinB1 = _mm_add_pd(_mm_sub_pd(E_I, mm_b), mm_min1);
            Tmp1 = _mm_mul_pd(kkMulE_I, E_IMinB1);
            Tmp2 = _mm_mul_pd(RTmp, mm_min1); // -R_tmp[i]
            Tmp3 = _mm_div_pd(M1MulR, E_IAddM2);
            Tmp4 = _mm_sub_pd(Tmp2, Tmp1);
            RTmp = _mm_add_pd(RTmp, _mm_mul_pd(mm_dt, _mm_mul_pd(_mm_add_pd(mm_eps, Tmp3), Tmp4)));
            _mm_storeu_pd(&R_tmp[i], RTmp);
        }
#else
        for (int i = 0; i < n; i++) {
	        E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
#endif
    }
#else
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
        E_prev_tmp = E_prev + j;

#ifdef SSE_VEC
        for(i = 0; i < n; i+=2) {
            __m128d E_I, E_T, E_B, E_R, E_L;
            __m128d Tmp1, Tmp2, Tmp3, Tmp4;
            __m128d ETmp_Tmp, ETmp;
            E_I = _mm_loadu_pd(&E_prev_tmp[i]);
            E_T = _mm_loadu_pd(&E_prev_tmp[i - (n + 2)]);
            E_B = _mm_loadu_pd(&E_prev_tmp[i + (n + 2)]);
            E_R = _mm_loadu_pd(&E_prev_tmp[i + 1]);
            E_L = _mm_loadu_pd(&E_prev_tmp[i - 1]);

            Tmp1 = _mm_add_pd(E_R, E_L);
            Tmp2 = _mm_add_pd(E_T, E_B);
            Tmp3 = _mm_mul_pd(E_I, mm_four);
            Tmp4 = _mm_sub_pd(_mm_add_pd(Tmp1, Tmp2), Tmp3);
            ETmp_Tmp = _mm_add_pd(E_I, _mm_mul_pd(mm_alph, Tmp4));
            _mm_storeu_pd(&E_tmp[i], ETmp_Tmp);
        }
#else
        for (i=0; i<n; i++) {
            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
        }
#endif
    }

    /* 
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
        R_tmp = R + j;
	E_prev_tmp = E_prev + j;
#ifdef SSE_VEC
        for(i = 0; i < n; i+=2) {
            __m128d E_I, ETmp, ETmp_Tmp, RTmp, E_IMinA, E_IMin1, E_IMulR;
            __m128d Tmp1, Tmp2, Tmp3, Tmp4, kkMulE_I, M1MulR, E_IAddM2, E_IMinB1;
            E_I = _mm_loadu_pd(&E_prev_tmp[i]);
            RTmp = _mm_loadu_pd(&R_tmp[i]);
            ETmp_Tmp = _mm_loadu_pd(&E_tmp[i]);

            // E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            kkMulE_I = _mm_mul_pd(mm_kk, E_I);
            E_IMinA = _mm_sub_pd(E_I, mm_A);
            E_IMin1 = _mm_add_pd(E_I, mm_min1);
            E_IMulR = _mm_mul_pd(E_I, RTmp);
            Tmp4 = _mm_mul_pd(E_IMinA, E_IMin1);
            ETmp = _mm_sub_pd(ETmp_Tmp, _mm_mul_pd(mm_dt, _mm_add_pd(_mm_mul_pd(kkMulE_I, Tmp4), E_IMulR)));
            _mm_storeu_pd(&E_tmp[i], ETmp);

            // R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
            M1MulR = _mm_mul_pd(mm_M1, RTmp);
            E_IAddM2 = _mm_add_pd(E_I, mm_M2);
            E_IMinB1 = _mm_add_pd(_mm_sub_pd(E_I, mm_b), mm_min1);
            Tmp1 = _mm_mul_pd(RTmp, mm_min1); // -R_tmp[i]
            Tmp2 = _mm_mul_pd(kkMulE_I, E_IMinB1);
            Tmp3 = _mm_div_pd(M1MulR, E_IAddM2);
            Tmp4 = _mm_sub_pd(Tmp1, Tmp2);
            RTmp = _mm_add_pd(RTmp, _mm_mul_pd(mm_dt, _mm_mul_pd(_mm_add_pd(mm_eps, Tmp3), Tmp4)));
            _mm_storeu_pd(&R_tmp[i], RTmp);
        }
#else 
        for (i = 0; i < n; i++) {           
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
#endif
    }
#endif
     /////////////////////////////////////////////////////////////////////////////////

    // // printf("Safe after A-P Model Inner\n");
    // if (cnt) MPI_Waitall(cnt, rec_requests, status);
    // // printf("Safe after MPI Waiting\n");
    // // Now Ghost cells are ready & Solve the rest
    // E_tmp = E + (n + 2) + 1;
    // for (i = 0; i < n; ++i) {
    //     E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
    //     E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
    //     R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
    // }
    // for (j = 1, i = 1; j < (m - 2); ++j) {
    //     E_tmp += j * (n + 2);
    //     E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
    //     E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
    //     R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
    // }
    // E_tmp = E + (n + 2) * m + 1;
    // for (i = 0; i < n; ++i) {
    //     E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
    //     E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
    //     R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
    // }
    // E_tmp = E + (n + 2) + n;
    // for (j = 1, i = n; j < (m - 2); ++j) {
    //     E_tmp += j * (n + 2);
    //     E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
    //     E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
    //     R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
    // }

   if (cb.stats_freq){
     if ( !(niter % cb.stats_freq)){
        stats(E,m,n,&mx,&sumSq);
        double l2norm = L2Norm(sumSq);
        repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
    }
   }

   if (cb.plot_freq){
          if (!(niter % cb.plot_freq)){
	    plotter->updatePlot(E,  niter, m, n);
        }
    }

   // Swap current and previous meshes
   double *tmp = E; E = E_prev; E_prev = tmp;

 } //end of 'niter' loop at the beginning

  //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters

    stats(E_prev,m,n,&Linf,&sumSq);
    MPI_Barrier(MPI_COMM_WORLD);
    // Calculate Linf / L2 by reduce
    MPI_Reduce(&Linf, &pLinf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sumSq, &pSumSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    Linf = pLinf;
    sumSq = pSumSq;
    L2 = L2Norm(sumSq);
    // MPI_Type_free(&MPI_COL_VEC);

  // Swap pointers so we can re-use the arrays
  *_E = E;
  *_E_prev = E_prev;
}

#endif // _MPI