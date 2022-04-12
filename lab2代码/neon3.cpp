#include<cstdlib>
#include<algorithm>
#include<iostream>
#include<ctime>
#include <arm_neon.h>
#include <sys/time.h>
using namespace std;

struct timeval time1,time2;
int matrix_to_coo(float **M,int n,float* &value,int* & row,int* & col){
    
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

double coo_multiply_vector_serial(int nonzeros,int n,int* row,int* col,float* value,float* x,float* y){
    //其中x指的是列向量，这里表示的是稀疏矩阵和列向量相乘
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
double coo_multiply_vector_neon(int nonzeros,int n,int* row,int* col,float* value,float* x,float* y){
    //__m128 t1,t3,sum;
    float32x4_t t1,t2,t3,sum;
    double timeuse;
    gettimeofday(&time1,NULL);
    int choice = nonzeros % 4;
    for(int i=0;i<n;i++)
        y[i]=0;
    for (int i=0;i<nonzeros-choice;i+=4){
        //t1 = _mm_set_ps(x[col[i+3]],x[col[i+2]],x[col[i+1]],x[col[i]]);
        sum=vdupq_n_f32(0.f);
        t1=vsetq_lane_f32(x[col[i+3]],t1,3);
        t1=vsetq_lane_f32(x[col[i+2]],t1,2);
        t1=vsetq_lane_f32(x[col[i+1]],t1,1);
        t1=vsetq_lane_f32(x[col[i]],t1,0);
        //vsetq_lane_f32((float32_t a, float32x2_t v, const int lane); //设置向量v第lane个通道的元
        t3 = vld1q_f32(value+i);
        sum = vmulq_f32(t3,t1);
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

void generate_vector(int n,float* & x){
    x=new float[n];
    for(int i=0;i<n;i++)
        x[i]=rand()%10+1;
}

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

void generate_sparse_matrix(float** & m,int n,double s){
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


int main()
{
    for(int p=1;p<10;p++){
        for(double k=0.002;k<0.01;k+=0.002){
            int n=1024*p;
            srand((int)time(0));
            double s=k;
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
            double neon_mul_sum = 0;
            for(int i=0;i<10;i++){
                serial_sum += coo_multiply_vector_serial(notzeros,n,row,col,value,vec,yy);
                neon_mul_sum += coo_multiply_vector_neon(notzeros,n,row,col,value,vec,yyy);
            }
            cout<<"规模为："<<n<<"稀疏度为："<<s<<" 平均耗时:"<<endl;
            cout<<"串行算法耗时:"<<serial_sum/10<<"s"<<endl;
            cout<<"neon版本耗时:"<<neon_mul_sum/10<<"s"<<endl;
        }
    }
    return 0;
}
