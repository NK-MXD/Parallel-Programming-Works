/**
 * @file mainCUDA.cpp
 * @author xiaoduo
 * @brief CUDA版本的COO格式的稀疏矩阵相乘
 * @version 0.1
 * @date 2022-06-15
 * @details 主要实现两个spMV和spMM
 * @copyright Copyright (c) 2022
 * 
 */
//下面是矩阵乘法的小例子:
#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}
//必要的函数
//生成稠密矩阵
void generate_matrix(float** & m,int n){
   m=new float*[n];
   for(int i=0;i<n;i++)
       m[i]=new float[n];
   for(int i=0;i<n;i++)
   {
       for(int j=0;j<n;j++)
      {
          int x=rand()%10;
          m[i][j]=x+1;
      }
   }
}

//生成稀疏矩阵
void generate_sparse_matrix(float** & m,int n,double s){
    //注：s为稀疏度
   m=new float*[n];
   for(int i=0;i<n;i++)
       m[i]=new float[n];
   for(int i=0;i<n;i++)
      for(int j=0;j<n;j++)
      {
          int x=rand()%1000;
          if(x>=1000*s)
            m[i][j]=0;
          else
            m[i][j]=x%10+1;
      }
   return;
}


int n = 4096;//矩阵规模
int size = n * sizeof(float);
int sizeM = n * n * sizeof (int); // Number of bytes of an N x N matrix
int nonzeros=0;//稀疏矩阵中非零元素的个数
int nozerorows=0;//稀疏矩阵中不全为0的行数，这个变量是未来进行稀疏矩阵pThread算法优化的关键变量
double s=0.005;

float **mat=NULL;//稀疏矩阵
float **mat_nonsparse=NULL;//稠密矩阵
float **mat_gpu=NULL;//结果矩阵1
float **mat_cpu=NULL;//结果矩阵2

float *vec=NULL;//向量
float *y_gpu=NULL;//spmv结果1
float *y_cpu=NULL;//spmv结果2

//稀疏矩阵表示法：在pThread编程中，为了方便起见，我们将所有行的首个元素的下标都存储在index数组中
float *value=NULL;
int *col=NULL;
int *row=NULL;
int *indexVec=NULL;

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}
//将稀疏矩阵转化为COO表示的格式
int matrix_to_coo(float **M,int n,float* &value,int* & row,int* & col,int* & index){
    //n为矩阵行列数 nonzeros代表矩阵的非零元素个数
   int i,j;
   int a=0;
   for(i=0;i<n;i++)
      for(j=0;j<n;j++)
          if(M[i][j]!=0)
              a++;
   checkCuda( cudaMallocManaged(&value, sizeof(float) * a) );
   checkCuda( cudaMallocManaged(&col, sizeof(int) * a) );
   checkCuda( cudaMallocManaged(&row, sizeof(int) * a) );
   int k=0;
   for(i=0;i<n;i++)
   {
      for(j=0;j<n;j++)
      {
          if(M[i][j]!=0)
          {
              row[k]=i;
              col[k]=j;
              value[k++]=M[i][j];
          }
      }
   }

   for(int k=1;k<a;k++){
      if(row[k]!=row[k-1]){
          nozerorows++;
      }
   }
   nozerorows=nozerorows+1;
   checkCuda( cudaMallocManaged(&indexVec, sizeof(int) * (nozerorows+1)) );
   int p=0;
   indexVec[p++]=0;
   for(int k=1;k<a;k++){
      if(row[k]!=row[k-1]){
          indexVec[p++]=k;
      }
   }
   indexVec[nozerorows]=nonzeros;//这里是一个哨兵
   return a;
}

//初始化
void init(){
    srand((int)time(0));
    s=0.005;
    checkCuda( cudaMallocManaged(&vec, size) );
    checkCuda( cudaMallocManaged(&y_cpu, size) );
    checkCuda( cudaMallocManaged(&y_gpu, size) );
    initWith(0, y_cpu, n);
    initWith(0, y_gpu, n);
    initWith(3, vec, n);

    generate_sparse_matrix(mat,n,s);//生成稀疏矩阵mat
    generate_matrix(mat_nonsparse,n);//生成稠密矩阵
    size_t pitch;
    cudaMallocPitch(&mat_gpu, &pitch, sizeof(float) * n, n);
    cudaMemset2D(mat_gpu, pitch, 0, sizeof(float) * n, n);
    cudaMallocPitch(&mat_cpu, &pitch, sizeof(float) * n, n);
    cudaMemset2D(mat_cpu, pitch, 0, sizeof(float) * n, n);
    nonzeros=matrix_to_coo(mat,n,value,row,col,indexVec);//生成对应的COO表示的稀疏矩阵
}

//实现COO和向量相乘（串行算法）
void spmv_cpu(){
    //其中x指的是列向量，这里表示的是稀疏矩阵和列向量相乘
    for(int i=0;i<nozerorows;i++)
    {
        for(int j=indexVec[i];j<indexVec[i+1];j++)
        {
            y_cpu[row[j]]+=value[j]*vec[col[i]];
        }
    }
}

//实现COO与稠密矩阵相乘串行算法
void spmm_cpu(){
    for (int i=0;i<nonzeros;i++)
        for(int k=0;k<n;k++)
            mat_cpu[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
}

__global__
void spmv_gpu(int nozerorows,float* vec,int* row,int* col,float* value,int* indexVec,float* y){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=index;i<nozerorows;i+=stride)
    {
        for(int j=indexVec[i];j<indexVec[i+1];j++)
        {
            y[row[j]]+=value[j]*vec[col[i]];
        }
    }
}

__global__
void spmm_gpu(int nonzeros,int n,int* row,int* col,float* value,float**b,float**c){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i=index;i<nonzeros;i++)
        for(int k=0;k<n;k++)
            c[row[i]][k] += value[i] * b[col[i]][k];
}

int main()
{
  init();
  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 128;
  numberOfBlocks = (nozerorows + threadsPerBlock - 1) / threadsPerBlock;
  cudaEvent_t start, stop;//计时器
  float elapsedTime = 0.0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);//开始计时
  //spmv_gpu<<<numberOfBlocks, threadsPerBlock>>>(nozerorows, vec,row, col, value,indexVec,y_gpu);
  spmm_gpu<<<numberOfBlocks, threadsPerBlock>>>(nozerorows,n,row, col, value,mat_nonsparse,mat_gpu);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);//停止计时
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("GPU_LU:%f ms\n", elapsedTime);

  checkCuda( cudaGetLastError() );
  checkCuda( cudaDeviceSynchronize() );
  cudaEventRecord(start, 0);//开始计时
  spmm_cpu();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);//停止计时
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("CPU_LU:%f ms\n", elapsedTime);


  // Free all our allocated memory
  checkCuda( cudaFree(value) );
  checkCuda( cudaFree(row) );
  checkCuda( cudaFree(col) );
  checkCuda( cudaFree(indexVec) );
  checkCuda( cudaFree(vec) );
  checkCuda( cudaFree(y_cpu) );
  checkCuda( cudaFree(y_gpu) );
  checkCuda( cudaFree(mat_gpu) );
  checkCuda( cudaFree(mat_cpu) );

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


