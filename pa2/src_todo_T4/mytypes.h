#ifndef MYTYPES_H
#define MYTYPES_H

// parameters according to CUTLASS blog
#define BLOCK_N 128
#define BLOCK_M 128
#define BLOCK_K 16
#define THREAD_M 8
#define THREAD_N 8
#define GRID_SIZE 256

// tilescale (# of points computed by each thread)
#ifndef TILESCALE_M
#define TILESCALE_M 1 // Enter your own values
#endif
#ifndef TILESCALE_N
#define TILESCALE_N 1 // Enter your own values
#endif
#ifndef TILESCALE_K
#define TILESCALE_K 1 // Enter your own values
#endif

#define TILEDIM_M 32 // Enter your own values
#define TILEDIM_N 32 // Enter your own values

// matrix A loads
// with warps along the horiziontal axis (K)
// so to get good coalescaed loads, we want TILEDIM_K to be >= 32
//
#define TILEDIM_K 4 // Enter your own values

// step size in each dimension
#define TILESTEP_N 1 // Enter your own values
#define TILESTEP_K 1 // Enter your own values
#define TILESTEP_M 1 // Enter your own values
 
#endif
