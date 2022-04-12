#include<cstdlib>
#include<algorithm>
#include<iostream>
#include<ctime>
#include <arm_neon.h>
using namespace std;

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

void coo_multiply_matrix_serial(int nonzeros,int n,int* row,int* col,float* value,float**b,float**c){

    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            c[i][j]=0;

    for (int i=0;i<nonzeros;i++)
        for(int k=0;k<n;k++)
            c[row[i]][k] += value[i] * b[col[i]][k];
}


void coo_multiply_matrix_neon(int nonzeros,int n,int* row,int* col,float* value,float**b,float**c){
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
    int n=4096;
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
    clock_t start,end;
    start=clock();
    coo_multiply_matrix_serial(notzeros,n,row,col,value,mat_nonsparse,mat_res1);
    end=clock();
    printf("time1=%f\n",(double)(end-start)/CLOCKS_PER_SEC);
    start=clock();
    coo_multiply_matrix_neon(notzeros,n,row,col,value,mat_nonsparse,mat_res2);
    end=clock();
    printf("time2=%f\n",(double)(end-start)/CLOCKS_PER_SEC);
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            if(mat_res1[i][j]!=mat_res2[i][j])
            {
                cout<<"test error!"<<endl;
                return -1;
            }
    cout<<"test right!"<<endl;
    return 0;

}









