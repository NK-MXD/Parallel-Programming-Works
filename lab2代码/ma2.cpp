#include<pmmintrin.h>
#include<xmmintrin.h>
#include<cstdlib>
#include<algorithm>
#include <sys/time.h>
#include<iostream>
#include<ctime>
#include <immintrin.h> //AVX、AVX2
using namespace std;

///要实现的几个函数功能：1.将矩阵转化为COO表示的稀疏矩阵 2.将COO表示的稀疏矩阵转化为矩阵 3.实现非COO表示的矩阵的相乘 4.实现COO表示的稀疏矩阵相乘
///5.实现COO表示的稀疏矩阵的并行乘法

struct timeval time1,time2;
///1.矩阵转化为COO表示的稀疏矩阵
int matrix_to_coo(float **M,int n,float* &value,int* & row,int* & col){
    //n为矩阵行列数 nonzeros代表矩阵的非零元素个数
   int i,j;
   int a=0;
   for(i=0;i<n;i++)
      for(j=0;j<n;j++)
          if(M[i][j]!=0)
              a++;
   value=new float[a];
   col=new int[a];
   row=new int[a];
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
   return a;
}
///2.将COO表示的稀疏矩阵转化为矩阵
void coo_to_matrix(float *value,int *col,int *row,int nonzeros,int n,float** & M){
   M=new float*[n];
   for(int i=0;i<n;i++)
      M[i]=new float[n];
   for(int i=0;i<n;i++)
       for(int j=0;j<n;j++)
           M[i][j]=0;
   for(int i=0;i<nonzeros;i++)
   {
       M[row[i]][col[i]]=value[i];
   }
}

///3.实现COO和向量相乘（串行算法）
double coo_multiply_vector_serial(int nonzeros,int n,int* row,int* col,float* value,float* x,float* y){
    double timeuse;
    gettimeofday(&time1,NULL);
    for(int i=0;i<n;i++)
        y[i]=0;
    for (int i=0;i<nonzeros;i++)
        y[row[i]] += value[i] * x[col[i]];//最后y中得出的结果是最后的稀疏矩阵和列向量相乘的结果
    gettimeofday(&time2,NULL);
    timeuse = (time2.tv_sec - time1.tv_sec) + (double)(time2.tv_usec - time1.tv_usec)/1000000.0;
    return timeuse;
}


///11.实现COO与向量相乘SIMD优化算法SSE1
double coo_multiply_vector_sse1(int nonzeros,int n,int* row,int* col,float* value,float* x,float* y){
    __m128 t1,t3,sum;
    double timeuse;
    gettimeofday(&time1,NULL);
    int choice = nonzeros % 4;
    for(int i=0;i<n;i++)
        y[i]=0;
    for (int i=0;i<nonzeros-choice;i+=4){
        t1 = _mm_set_ps(x[col[i+3]],x[col[i+2]],x[col[i+1]],x[col[i]]);
        sum = _mm_setzero_ps();
        t3 = _mm_load_ps(value+i);
        sum = _mm_mul_ps(t3,t1);
        for(int j=0;j<4;j++)
            y[row[i+j]] += sum[j];
    }
    //对齐操作
    for(int i=nonzeros - choice; i < nonzeros ; i++){
        y[row[i]] += value[i] * x[col[i]];
    }
    gettimeofday(&time2,NULL);
    timeuse = (time2.tv_sec - time1.tv_sec) + (double)(time2.tv_usec - time1.tv_usec)/1000000.0;
    return timeuse;
}

///12.实现COO与向量相乘SIMD优化算法SSE2

///6.生成向量
void generate_vector(int n,float* & x){
    x=new float[n];
    for(int i=0;i<n;i++)
        x[i]=rand()%10+1;
}

///7.生成稠密矩阵
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


///8.生成稀疏矩阵
void generate_sparse_matrix(float** & m,int n,double s){
    //注：s为稀疏度
   m=new float*[n];
   for(int i=0;i<n;i++)
       m[i]=new float[n];
   for(int i=0;i<n;i++)
      for(int j=0;j<n;j++)
      {
          int x=rand()%100000;
          if(x>100000*s)
            m[i][j]=0;
          else
            m[i][j]=x%10+1;
      }
   return;
}

///主函数测试：
int main()
{
    for(int p=1;p<10;p++){
        int n=1024*p;
        srand((int)time(0));
        double s=0.008;
        float **mat=NULL;
        float **mat_recover=NULL;
        float **mat_nonsparse=NULL;
        float **mat_res1=NULL;
        float **mat_res2=NULL;
        float **mat_res3=NULL;
        float *vec=NULL;
        float *y=NULL;
        float *yy=NULL;
        float *yyy=NULL;
        float *value=NULL;
        int *col=NULL;
        int *row=NULL;
        generate_vector(n,vec);
        generate_vector(n,y);
        generate_vector(n,yy);
        generate_vector(n,yyy);
        generate_sparse_matrix(mat,n,s);//生成稀疏矩阵mat
        generate_matrix(mat_nonsparse,n);
        generate_matrix(mat_res1,n);
        generate_matrix(mat_res2,n);
        generate_matrix(mat_res3,n);
        int notzeros=matrix_to_coo(mat,n,value,row,col);//生成对应的COO表示的稀疏矩阵
        double serial_sum = 0;
        double sse_mul_sum = 0;
        for(int i=0;i<20;i++){
            serial_sum += coo_multiply_vector_serial(notzeros,n,row,col,value,vec,yy);
            sse_mul_sum += coo_multiply_vector_sse1(notzeros,n,row,col,value,vec,yyy);
        }
        cout<<"规模为："<<n<<"稀疏度为："<<s<<" 平均耗时:"<<endl;
        cout<<"串行算法耗时:"<<serial_sum/20<<"s"<<endl;
        cout<<"SSE版本耗时:"<<sse_mul_sum/20<<"s"<<endl;
    }

    return 0;

}









