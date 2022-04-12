#include<pmmintrin.h>
#include<xmmintrin.h>
#include<cstdlib>
#include<algorithm>
#include<windows.h>
#include<iostream>
#include<ctime>
#include <immintrin.h> //AVX、AVX2
using namespace std;

///要实现的几个函数功能：1.将矩阵转化为COO表示的稀疏矩阵 2.将COO表示的稀疏矩阵转化为矩阵 3.实现非COO表示的矩阵的相乘 4.实现COO表示的稀疏矩阵相乘
///5.实现COO表示的稀疏矩阵的并行乘法
int maxN=1024;
const int T=64;

long long head,tail,freq;


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
void coo_multiply_vector_serial(int nonzeros,int n,int* row,int* col,float* value,float* x,float* y){
    //其中x指的是列向量，这里表示的是稀疏矩阵和列向量相乘
    __m128 t1,t2,sum;
    for(int i=0;i<n;i++)
        y[i]=0;
    for (int i=0;i<nonzeros;i++)
        y[row[i]] += value[i] * x[col[i]];//最后y中得出的结果是最后的稀疏矩阵和列向量相乘的结果
}

///4.实现矩阵和向量相乘（平凡算法）
void matrix_multiply_vector_serial(float **m,int n,float *x,float *y){
   for(int i=0;i<n;i++)
   {
       int y0=0;
       for(int j=0;j<n;j++)
           y0+=m[i][j]*x[j];
       y[i]=y0;
   }
}

///11.实现COO与向量相乘SIMD优化算法SSE1
void coo_multiply_vector_sse1(int nonzeros,int n,int* row,int* col,float* value,float* x,float* y){
    __m128 t1,t3,sum;
    int choice = nonzeros % 4;
    //并行算法
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
}

///12.实现COO与向量相乘SIMD优化算法SSE2

///5.稀疏矩阵和稠密矩阵相乘SIMD算法优化到最优,a或b的其中一个为稀疏矩阵，一个为稠密矩阵
double matrix_multiply_matrix_sse_title(int n, float**a, float**b,float**c){
    __m128 t1,t2,sum;
    float t;
    long long head,tail,freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
    for (int r = 0; r < n / T; ++r) for (int q = 0; q < n / T; ++q) {
        for (int i = 0; i < T; ++i) for (int j = 0; j < T; ++j) c[r * T + i][q * T + j] = 0.0;
        for (int p = 0; p < n / T; ++p) {
            for (int i = 0; i < T; ++i) for (int j = 0; j < T; ++j) {
                sum = _mm_setzero_ps();
                for (int k = 0; k < T; k += 4){
                    t1 = _mm_loadu_ps(a[r * T + i] + p * T + k);
                    t2 = _mm_loadu_ps(b[q * T + j] + p * T + k);
                    t1 = _mm_mul_ps(t1, t2);
                    sum = _mm_add_ps(sum, t1);
                }
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);
                _mm_store_ss(&t, sum);
                c[r * T + i][q * T + j] += t;
            }
        }
    }
    for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout<<"sse_tile:"<<(double)(tail-head)/(double)freq<<endl;
	return (double)(tail-head)/(double)freq;
}
///9.实现COO与稠密矩阵相乘串行算法
double coo_multiply_matrix_serial(int nonzeros,int n,int* row,int* col,float* value,float**b,float**c){
    long long head,tail,freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            c[i][j]=0;

    for (int i=0;i<nonzeros;i++)
        for(int k=0;k<n;k++)
            c[row[i]][k] += value[i] * b[col[i]][k];

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout<<"sse1:"<<(double)(tail-head)/(double)freq<<endl;
	return (double)(tail-head)/(double)freq;
}


///10.实现COO与稠密矩阵相乘并行算法SEE1
///基本思路是将列向量相乘进行并行化处理
double coo_multiply_matrix_sse1(int nonzeros,int n,int* row,int* col,float* value,float**b,float**c){
    __m128 t1,t3,sum;
    long long head,tail,freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    int choice = nonzeros % 4;
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            c[i][j]=0;

    for(int k=0;k<n;k++)
    {
        for (int i=0;i<nonzeros-choice;i+=4)
        {
            t1 = _mm_set_ps(b[col[i+3]][k],b[col[i+2]][k],b[col[i+1]][k],b[col[i]][k]);
            sum = _mm_setzero_ps();
            t3 = _mm_load_ps(value+i);
            sum = _mm_mul_ps(t3,t1);
            //加了这个循环使得并行的效果被掩盖了
            for(int j=0;j<4;j++)
            {
                c[row[i+j]][k] += sum[j];
            }
        }
        for(int i=nonzeros-choice;i < nonzeros;i++){
            c[row[i]][k] += value[i] * b[col[i]][k];
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout<<"sse1:"<<(double)(tail-head)/(double)freq<<endl;
	return (double)(tail-head)/(double)freq;
}

///13.实现COO与稠密矩阵相乘并行算法SEE2
///基本思路是循环展开
double coo_multiply_matrix_sse2(int nonzeros,int n,int* row,int* col,float* value,float**b,float**c){
    long long head,tail,freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    __m128 t1,t2,t3,sum;
    int choice = n % 4;
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            c[i][j]=0;

    for (int i=0;i<nonzeros;i++)
    {
        for(int k=0;k<n-choice;k+=4)
        {
            t1=_mm_load_ps(b[col[i]]+k);
            sum = _mm_setzero_ps();
            t3 = _mm_set_ps1(value[i]);
            t2=_mm_load_ps(c[row[i]]+k);
            sum = _mm_mul_ps(t3,t1);
            t2=_mm_add_ps(t2,sum);
            _mm_store_ps(c[row[i]]+k,t2);
        }
        for(int k=n-choice;k < n;k++){
            c[row[i]][k] += value[i] * b[col[i]][k];
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout<<"sse2:"<<(double)(tail-head)/(double)freq<<endl;
	return (double)(tail-head)/(double)freq;
}

///14.实现COO与稠密矩阵相乘并行算法SEE3
///这种方法的思路是先进行矩阵的分块然后进行分块矩阵的乘法的并行优化:不会了

///15.实现COO与稠密矩阵相乘的并行算法AVX
///实现原理是实现八路并行和十六路并行
void coo_multiply_matrix_avx(int nonzeros,int n,int* row,int* col,float* value,float**b,float**c){
     __m512 _t1, _t2, _t3, _sum;
    int choice = n % 16;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            c[i][j] = 0;

    for (int i = 0; i < nonzeros; i++)
    {
        for (int k = 0; k < n - choice; k += 16)
        {
            _sum = _mm512_setzero_ps();
            _t1 = _mm512_load_ps(b[col[i]] + k);
            _t2 = _mm512_load_ps(c[row[i]] + k);
            _t3 = _mm512_set1_ps(value[i]);
            _sum = _mm512_mul_ps(_t3, _t1);
            _t2 = _mm512_add_ps(_t2, _sum);
            _mm512_store_ps(c[row[i]]+k,_t2);
        }
        for (int k = n - choice; k < n; k++) {
            c[row[i]][k] += value[i] * b[col[i]][k];
        }
    }
}

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
    int n=1024;
    srand((int)time(0));
    double s=0.002;
    float **mat=NULL;
    float **mat_nonsparse=NULL;
    float **mat_res1=NULL;
    float **mat_res2=NULL;
    float **mat_res3=NULL;
    float *value=NULL;
    int *col=NULL;
    int *row=NULL;
    generate_sparse_matrix(mat,n,s);//生成稀疏矩阵mat
    generate_matrix(mat_nonsparse,n);
    generate_matrix(mat_res1,n);
    generate_matrix(mat_res2,n);
    generate_matrix(mat_res3,n);
    int notzeros=matrix_to_coo(mat,n,value,row,col);//生成对应的COO表示的稀疏矩阵
    ///以下为测试稀疏矩阵和矩阵相乘结果是否正确
    clock_t start,end;
    start=clock();
    coo_multiply_matrix_serial(notzeros,n,row,col,value,mat_nonsparse,mat_res1);
    end=clock();
    printf("平凡算法time1=%f\n",(double)(end-start)/CLK_TCK);
    start=clock();
    coo_multiply_matrix_sse2(notzeros,n,row,col,value,mat_nonsparse,mat_res3);
    end=clock();
    printf("SSE优化2time2=%f\n",(double)(end-start)/CLK_TCK);
    start=clock();
    coo_multiply_matrix_avx(notzeros,n,row,col,value,mat_nonsparse,mat_res2);
    end=clock();
    printf("AVX优化1time3=%f\n",(double)(end-start)/CLK_TCK);
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            if(mat_res3[i][j]!=mat_res2[i][j])
            {
                cout<<"test error!"<<endl;
                return -1;
            }
    cout<<"test right!"<<endl;
    return 0;

}









