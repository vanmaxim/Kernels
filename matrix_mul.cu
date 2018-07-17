#include <cuda.h>
#include <stdio.h>

int const TILE_WIDTH = 2;

__global__
void TiledMatrixMulKernel(float *M, float *N, float *P, int Width)
{
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
  
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float Pvalue = 0;
  for (int ph = 0; ph < ceil(Width/(float)TILE_WIDTH); ++ph) {
    if ((Row < Width) && (ph * TILE_WIDTH + tx) < Width)
      Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
    
    if ((ph * TILE_WIDTH + ty) < Width && Col < Width)
      Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];

    __syncthreads();
  
    for (int k = 0; k < TILE_WIDTH; ++k) {
      Pvalue += Mds[ty][k] * Nds[k][tx];
   }
   __syncthreads();
  }

  if ((Row < Width) && (Col < Width))
    P[Row * Width + Col] = Pvalue;
}

__global__
void MatrixMulKernel(float *M, float *N, float *P, int Width)
{
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((Row < Width) && (Col < Width)) {
    float Pvalue = 0;
    for (int k = 0; k < Width; ++k) {
      Pvalue += M[Row * Width + k] * N[k * Width + Col];
    }
    P[Row * Width + Col] = Pvalue; 
  }
}

int main(void)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int N = 4 * 4; // 2x2 matrix
  int size = sizeof(float) * N;
  float *h_M = (float *)malloc(size);
  float *h_N = (float *)malloc(size);

  for (int i = 0; i < N; ++i) {
    h_M[i] = (float)i;
    h_N[i] = (float)i; 
  }

  float *h_P = (float *)malloc(size);

  float *d_M, *d_N, *d_P;
  cudaMalloc((void **)&d_M, size);
  cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_N, size);
  cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_P, size);

  dim3 dimGrid(2, 2, 1);
  dim3 dimBlock(2, 2, 1);

  printf("Tiled:\n");

  cudaEventRecord(start);
  TiledMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, 4);  
  cudaEventRecord(stop);

  cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_P);

  for (int i = 0; i < N; ++i)
    printf("%f\n", h_P[i]);

  printf("Time: %f\n", milliseconds);


  free(h_M);
  free(h_N);
  free(h_P);
}
