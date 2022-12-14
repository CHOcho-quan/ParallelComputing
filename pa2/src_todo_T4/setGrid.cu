
#include "mytypes.h"
#include <stdio.h>

void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{

   // set your block dimensions and grid dimensions here
   blockDim.x = GRID_SIZE;
   blockDim.y = blockDim.z = 1;
   gridDim.x = n / BLOCK_N;
   gridDim.y = n / BLOCK_M;
   gridDim.z = 1;

   // you can overwrite blockDim here if you like.
   if (n % BLOCK_N != 0)
      gridDim.x++;
   if (n % BLOCK_M != 0)
      gridDim.y++;
}
