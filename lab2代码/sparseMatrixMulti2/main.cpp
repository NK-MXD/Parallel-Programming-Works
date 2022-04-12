#include<iostream>
#include<cstdlib>
#include<ctime>
#include<stdio.h>
typedef int dtype;
using namespace std;

void csr_to_matrix(dtype *value,dtype *colindex,dtype *rowptr,int n,int a,dtype** & M){
   M=new int*[n];
   for(int i=0;i<n;i++)
      M[i]=new int[n];
   for(int i=0;i<n;i++)
       for(int j=0;j<n;j++)
           M[i][j]=0;
   for(int i=0;i<n;i++)
       for(int j=rowptr[i];j<rowptr[i+1];j++)
           M[i][colindex[j]]=value[j];
   return;
}

void spmv(dtype *value,dtype *rowptr,dtype *colindex,int n,int a,dtype *x,dtype *y){
    //calculate the matrix-vector multiply where matrix is stored in the form of CSR
    for(int i=0;i<n;i++){
        dtype y0=0;
        for(int j=rowptr[i];j<rowptr[i+1];j++)
            y0+=value[j]*x[colindex[j]];
        y[i]=y0;
    }
    return;
}

int matrix_to_csr(int n,dtype **M,dtype* &value,dtype* & rowptr,dtype* & colindex){
   int i,j;
   int a=0;
   for(i=0;i<n;i++)
      for(j=0;j<n;j++)
          if(M[i][j]!=0)
              a++;
   value=new dtype[a];
   colindex=new int[a];
   rowptr=new int[n+1];
   int k=0;
   int l=0;
   for(i=0;i<n;i++)
      for(j=0;j<n;j++){
          if(j==0)
              rowptr[l++]=k;
          if(M[i][j]!=0){
              value[k]=M[i][j];
              colindex[k]=j;
              k++;}
   }
   rowptr[l]=a;
   return a;
}

void matrix_multiply_vector(dtype **m,int n,dtype *x,dtype *y){
   for(int i=0;i<n;i++)
   {
       dtype y0=0;
       for(int j=0;j<n;j++)
           y0+=m[i][j]*x[j];
       y[i]=y0;
   }
   return;
}

void generate_sparse_matrix(dtype** & m,int n,double s){
   m=new int*[n];
   for(int i=0;i<n;i++)
       m[i]=new int[n];
   for(int i=0;i<n;i++)
      for(int j=0;j<n;j++)
      {
          int x=rand()%100;
          if(x>100*s)
            m[i][j]=0;
          else
            m[i][j]=x+1;
      }
   return;
}

void print_matrix(dtype **m,int n){
   for(int i=0;i<n;i++)
       for(int j=0;j<n;j++)
   {
           cout<<m[i][j]<<",";
           if(j==n-1)
               cout<<endl;
   }
   return;
}

void generate_vector(int n,dtype* & x){
    x=new int[n];
    for(int i=0;i<n;i++)
        x[i]=rand()%100-50;
    return;
}

void print_vector(int n,dtype* x){
    for(int i=0;i<n;i++)
        cout<<x[i]<<" ";
    return;
}
int main(){
    srand(time(0));
    int n=1024;
    double s=0.01;
    dtype **mat=NULL;
    dtype **mat_recover=NULL;
    dtype *vec=NULL;
    dtype *y=NULL;
    dtype *yy=NULL;
    dtype *value=NULL;
    int *colindex=NULL;
    int *rowptr=NULL;
    generate_sparse_matrix(mat,n,s);
    generate_vector(n,vec);
    generate_vector(n,y);
    generate_vector(n,yy);
    int a=matrix_to_csr(n,mat,value,rowptr,colindex);
    /*csr_to_matrix(value,colindex,rowptr,n,a,mat_recover);
    cout<<"matrix and csr transformation test"<<endl;
    int error=0;
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
           if(mat[i][j]!=mat_recover[i][j])
               error=1;
    if(error==1)
        cout<<"test error!"<<endl;
    else
        cout<<"test right!"<<endl;*/

    cout<<"spvm test"<<endl;
    clock_t start,end;
    start=clock();
    matrix_multiply_vector(mat,n,vec,y);
    end=clock();
    printf("time1=%f\n",(double)(end-start)/CLK_TCK);
    start=clock();
    spmv(value,rowptr,colindex,n,a,vec,yy);
    end=clock();
    printf("time2=%f\n",(double)(end-start)/CLK_TCK);
    for(int i=0;i<n;i++)
        if(y[i]!=yy[i])
    {
            cout<<"test error!"<<endl;
            return -1;
    }
    cout<<"test right!"<<endl;
    return 0;
}


