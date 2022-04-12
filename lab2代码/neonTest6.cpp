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

double coo_multiply_matrix_serial(int nonzeros,int n,int* row,int* col,float* value,float**b,float**c){
    double timeuse;
    gettimeofday(&time1,NULL);
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            c[i][j]=0;

    for (int i=0;i<nonzeros;i++)
        for(int k=0;k<n;k++)
            c[row[i]][k] += value[i] * b[col[i]][k];
    gettimeofday(&time2,NULL);
    timeuse = (time2.tv_sec - time1.tv_sec) + (double)(time2.tv_usec - time1.tv_usec)/1000000.0;
    return timeuse;
}

//这是实现4路并行
double coo_multiply_matrix_neon4(int nonzeros,int n,int* row,int* col,float* value,float**b,float**c){
    double timeuse;
    gettimeofday(&time1,NULL);
    float32x4_t t1,t2,t3,sum;
    int choice = n % 4;
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            c[i][j]=0;

    for (int i=0;i<nonzeros;i++)
    {
        for(int k=0;k<n-choice;k+=4)
        {
            sum=vdupq_n_f32(0.f);
            t1=vld1q_f32(b[col[i]]+k);
            t2=vld1q_f32(c[row[i]]+k);
            t3 = vdupq_n_f32(value[i]);
            sum = vmulq_f32(t3,t1);
            t2= vaddq_f32(t2,sum);
            vst1q_f32(c[row[i]]+k,t2);
        }
        for(int k=n-choice;k < n;k++){
            c[row[i]][k] += value[i] * b[col[i]][k];
        }
    }
    gettimeofday(&time2,NULL);
    timeuse = (time2.tv_sec - time1.tv_sec) + (double)(time2.tv_usec - time1.tv_usec)/1000000.0;
    return timeuse;
}

//这是实现8路并行
/*void coo_multiply_matrix_neon8(int nonzeros,int n,int* row,int* col,float* value,float**b,float**c){
    float32x4x2_t t1,t2,t3,sum;
    int choice = n % 4;
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            c[i][j]=0;

    for (int i=0;i<nonzeros;i++)
    {
        for(int k=0;k<n-choice;k+=4)
        {
            sum=vdupq_n_f32(0.f);//这里如何加载？？？？vld2q_dup_f32
            t1=vld1q_f32_x2(b[col[i]]+k);//这里load  vld1q_f32_x2
            t2=vld1q_f32_x2(c[row[i]]+k);//
            t3 = vdupq_n_f32(value[i]);
            sum = vmulq_f32(t3,t1);//水平相乘
            t2= vaddq_f32_x2(t2,sum);//这里相加
            vst1q_f32(c[row[i]]+k,t2);//重新load
        }
        for(int k=n-choice;k < n;k++){
            c[row[i]][k] += value[i] * b[col[i]][k];
        }
    }
}*/

//这是实现16路并行
/*void coo_multiply_matrix_neon16(int nonzeros,int n,int* row,int* col,float* value,float**b,float**c){
    float32x4_t t1,t2,t3,sum;
    int choice = n % 4;
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            c[i][j]=0;

    for (int i=0;i<nonzeros;i++)
    {
        for(int k=0;k<n-choice;k+=4)
        {
            sum=vdupq_n_f32(0.f);
            t1=vld1q_f32(b[col[i]]+k);
            t2=vld1q_f32(c[row[i]]+k);
            t3 = vdupq_n_f32(value[i]);
            sum = vmulq_f32(t3,t1);
            t2= vaddq_f32(t2,sum);
            vst1q_f32(c[row[i]]+k,t2);
        }
        for(int k=n-choice;k < n;k++){
            c[row[i]][k] += value[i] * b[col[i]][k];
        }
    }
}*/

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
    for(int p=0;p<10;p++)
    {
        //for (double q = 0.001; q < 0.01; q += 0.001) {
            int n=1024*p;
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
            generate_sparse_matrix(mat,n,s);
            generate_matrix(mat_nonsparse,n);
            generate_matrix(mat_res1,n);
            generate_matrix(mat_res2,n);
            generate_matrix(mat_res3,n);
            int notzeros=matrix_to_coo(mat,n,value,row,col);
            double serial_mul_sum = 0;
            double neon_mul_sum = 0;
            
        
            serial_mul_sum += coo_multiply_matrix_serial(notzeros, n, row, col, value, mat_nonsparse, mat_res1);
            neon_mul_sum += coo_multiply_matrix_neon4(notzeros, n, row, col, value, mat_nonsparse, mat_res2);
            
            cout << "矩阵规模为: " << n <<"稀疏度为："<<s<< " 平均耗时:" << endl;
            cout << "平凡算法耗时:" << serial_mul_sum  << "s" << endl;
            cout << "Neon优化耗时:" << neon_mul_sum << "s" << endl;
        
    }
    return 0;
}





