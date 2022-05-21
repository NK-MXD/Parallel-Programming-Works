/**
-----------------------------------------------------------------------
本次实验是实现openMP并行的稀疏矩阵COO格式的乘法的运算,写一些优化思路要求：

1. openMP加进来，运行成功；
2. 重新整理程序，使得程序执行更加有条理，参数也调节更加方便；
3. 修改openMP中的线程数目，探索不同线程之间的执行时间的差别；
4. 整理所有并行思路的所有的运行时间，进行对比分析，存储在文件当中
5. 探索鲲鹏架构下程序的最优性能；

------------------------------------------------------------------------
linux版
arm 鲲鹏
------------------------------------------------------------------------
**/
#include<cstdlib>
#include<algorithm>
#include <sys/time.h>
#include<iostream>
#include<ctime>
#include<pthread.h>
#include <fstream>
#include <sstream>
#include <arm_neon.h>
#include<omp.h>
using namespace std;

///-------------------------------------------------------------一堆堆变量的定义----------------------------------------------
//一些变量进行定义：全局变量
const double Converter = 1000 * 1000; // 1s == 1000 * 1000 us

typedef struct{
	int	threadId;
} threadParm_t;

typedef struct{
	int	threadId;
	int rowid;
} threadParm_t2;

int n = 4096;//矩阵规模
int THREAD_NUM=0;//线程的个数
int OMP_NUM_THREADS=16;//openMP需要用到的线程的数目
int nonzeros=0;//稀疏矩阵中非零元素的个数
int nozerorows=0;//稀疏矩阵中不全为0的行数，这个变量是未来进行稀疏矩阵pThread算法优化的关键变量
int single_circle=20;//单个线程的工作量

float **mat=NULL;//稀疏矩阵
float **mat_nonsparse=NULL;//稠密矩阵
float **mat_res1=NULL;//结果矩阵1
float **mat_res2=NULL;//结果矩阵2
float **mat_res3=NULL;//结果矩阵2
float *vec=NULL;//向量
float *y=NULL;//spmv结果1
float *yy=NULL;//spmv结果2
float *yyy=NULL;//spmv结果3

//稀疏矩阵表示法：在pThread编程中，为了方便起见，我们将所有行的首个元素的下标都存储在index数组中
float *value=NULL;
int *col=NULL;
int *row=NULL;
int *myindex=NULL;

//这里是pthread+openMP需要用到的变量
int next_arr = 0;//这里是控制向量计算的公共的指针变量，需要在每一次计算之前进行归零
int next_arr2 = 0;//想办法统一一下
pthread_mutex_t  mutex_task;


///---------------------------------------------------------一堆堆通用的函数-----------------------------------------------------
//通用的一些算法
//生成向量
void generate_vector(int n,float* & x){
    x=new float[n];
    for(int i=0;i<n;i++)
        x[i]=rand()%10+1;
}

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

//将稀疏矩阵转化为COO表示的格式
int matrix_to_coo(float **M,int n,float* &value,int* & row,int* & col,int* & myindex){
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

   for(int k=1;k<a;k++){
      if(row[k]!=row[k-1]){
          nozerorows++;
      }
   }
   nozerorows=nozerorows+1;
   myindex=new int[nozerorows+1];
   int p=0;
   myindex[p++]=0;
   for(int k=1;k<a;k++){
      if(row[k]!=row[k-1]){
          myindex[p++]=k;
      }
   }
   myindex[nozerorows]=nonzeros;//这里是一个哨兵
   return a;
}

//实现COO和向量相乘（串行算法）
double coo_multiply_vector_serial(int nonzeros,int n,int* row,int* col,float* value,float* x,float* y){
    //其中x指的是列向量，这里表示的是稀疏矩阵和列向量相乘
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    for (int i=0;i<nonzeros;i++)
        y[row[i]] += value[i] * x[col[i]];//最后y中得出的结果是最后的稀疏矩阵和列向量相乘的结果
    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

//实现COO与稠密矩阵相乘串行算法
double coo_multiply_matrix_serial(){
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    for (int i=0;i<nonzeros;i++)
        for(int k=0;k<n;k++)
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];

    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

//实现COO与稠密矩阵相乘并行算法neon
///基本思路是循环展开，这里使用的是128位向量
double coo_multiply_matrix_neon(){
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    float32x4_t t1,t2,t3,sum;
    int choice = n % 4;
    for (int i=0;i<nonzeros;i++)
    {
        for(int k=0;k<n-choice;k+=4)
        {
            t1=vld1q_f32(mat_nonsparse[col[i]]+k);
            sum = vdupq_n_f32(0.f);
            t3 = vdupq_n_f32(value[i]);
            t2=vld1q_f32(mat_res1[row[i]]+k);
            sum = vmulq_f32(t3,t1);
            t2=vaddq_f32(t2,sum);
            vst1q_f32(mat_res1[row[i]]+k,t2);
        }
        for(int k=n-choice;k < n;k++){
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
        }
    }
    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

//初始化向量
void init_vector(){
    nonzeros=0;
    nozerorows=0;
    THREAD_NUM=4;
    srand((int)time(0));
    double s=0.005;
    y=new float[n]{0};
    yy=new float[n]{0};
    yyy=new float[n]{0};
    generate_vector(n,vec);//生成向量
    generate_sparse_matrix(mat,n,s);//生成稀疏矩阵mat
    nonzeros=matrix_to_coo(mat,n,value,row,col,myindex);//生成对应的COO表示的稀疏矩阵
    single_circle=nozerorows/(THREAD_NUM*100);//对于动态线程来讲，各种线程分配区别相差不大，这里以这个数据算出的结果为例
}
//初始化
void init(){
    nonzeros=0;
    nozerorows=0;
    THREAD_NUM=4;
    srand((int)time(0));
    double s=0.005;
    generate_vector(n,vec);//生成向量
    generate_sparse_matrix(mat,n,s);//生成稀疏矩阵mat
    generate_matrix(mat_nonsparse,n);//生成稠密矩阵
    mat_res1=new float*[n];
    mat_res2=new float*[n];
    for(int i=0;i<n;i++)
    {
        mat_res1[i]=new float[n]{0};
        mat_res2[i]=new float[n]{0};
    }
    nonzeros=matrix_to_coo(mat,n,value,row,col,myindex);//生成对应的COO表示的稀疏矩阵
    single_circle=nozerorows/(THREAD_NUM*100);//对于动态线程来讲，各种线程分配区别相差不大，这里以这个数据算出的结果为例
}

//----------------------------------------------------------------一堆堆子线程要干的事-------------------------------------------
//实现pThread的spmv算法1:静态线程分配
void* coo_multiply_vector_pthread1(void *parm){
    threadParm_t *p = (threadParm_t *) parm;
    int id = p->threadId;
    int seg=nozerorows/THREAD_NUM;
    for(int i=myindex[seg*id];i<myindex[seg*(id+1)];i++){
        yy[row[i]] += value[i] * vec[col[i]];
    }
    pthread_exit(nullptr);
}


//实现pThread的spmv算法2：动态线程分配 ---------这里的难点在于粒度的划分，4个线程，可以每十行进行变量的划分
void* coo_multiply_vector_pthread2(void *parm){
    threadParm_t *p = (threadParm_t *) parm;
    int id = p->threadId;
    int task = 0;

    while(1){
        pthread_mutex_lock(&mutex_task);
        task = next_arr++;
        pthread_mutex_unlock(&mutex_task);
        if (task >= nozerorows) break;
        for(int i=myindex[task];i<myindex[task+1];i++){
            yyy[row[i]] += value[i] * vec[col[i]];
        }
    }
    pthread_exit(NULL);
}

//实现pThread的spmm算法
///这里线程有两种划分模式，一种是直接在外层进行划分，另一种是在内层进行划分
///第一种实现的是在外层进行划分
void* coo_multiply_matrix_pthread1(void *parm){
    threadParm_t *p = (threadParm_t *) parm;
    int id = p->threadId;
    int interval=nozerorows/THREAD_NUM;
    int maxx=0;
    if(id==3){
        maxx=nonzeros;

    }else{
        maxx=myindex[interval*(id+1)];
    }

    for(int i=myindex[interval*id];i<maxx;i++){//计算的index[]是从row的i行到i+interval行
        for(int k=0;k<n;k++)
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
    }
    pthread_exit(NULL);
}

//这种算法开销太大了。。。。。。。
void* coo_multiply_matrix_pthread2(void *parm){
    threadParm_t2 *p = (threadParm_t2 *) parm;
    int id = p->threadId;
    int i=p->rowid;
    int interval=n/THREAD_NUM;
    int maxx=0;
    if(id==3){
        maxx=n;

    }else{
        maxx=interval*(id+1);
    }

    //for(int i=myindex[interval*id];i<maxx;i++){//计算的index[]是从row的i行到i+interval行
    for(int k=interval*id;k<maxx;k++)
        mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
    //}
    pthread_exit(NULL);
}

//实现pThread编程的spmm动态线程分配,+neno
void* coo_multiply_matrix_pthread4(void *parm){
    threadParm_t *p = (threadParm_t *) parm;
    int id = p->threadId;
    float32x4_t t1,t2,t3,sum;
    int choice = n % 4;
    int task = 0;
    int maxx;
    while(1){
        pthread_mutex_lock(&mutex_task);
        task = next_arr2;
        next_arr2+=single_circle;
        pthread_mutex_unlock(&mutex_task);
        if (task >= nozerorows) break;
        if(task>=nozerorows-single_circle)maxx=nonzeros;
        else maxx=myindex[task+single_circle];
        for(int i=myindex[task];i<maxx;i++){
            for(int k=0;k<n-choice;k+=4)
            {
                t1=vld1q_f32(mat_nonsparse[col[i]]+k);
                sum = vdupq_n_f32(0.f);
                t3 = vdupq_n_f32(value[i]);
                t2=vld1q_f32(mat_res1[row[i]]+k);
                sum = vmulq_f32(t3,t1);
                t2=vaddq_f32(t2,sum);
                vst1q_f32(mat_res1[row[i]]+k,t2);
            }
            for(int k=n-choice;k < n;k++){
                mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
            }
        }
    }
    pthread_exit(NULL);
}


//实现pThread编程的spmm动态线程分配2,不加neon
void* coo_multiply_matrix_pthread3(void *parm){
    threadParm_t *p = (threadParm_t *) parm;
    int id = p->threadId;
    int task = 0;
    int maxx;
    while(1){
        pthread_mutex_lock(&mutex_task);
        task = next_arr2;
        next_arr2+=single_circle;
        pthread_mutex_unlock(&mutex_task);
        if (task >= nozerorows) break;
        if(task>=nozerorows-single_circle)maxx=nonzeros;
        else maxx=myindex[task+single_circle];
        for(int i=myindex[task];i<maxx;i++){
            for(int k=0;k<n;k++)
                mat_res2[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
        }
    }
    pthread_exit(NULL);
}

///实现pThread编程，静态加neon
void* coo_multiply_matrix_pthread_sse1(void *parm){
    threadParm_t *p = (threadParm_t *) parm;
    int id = p->threadId;
    int interval=nozerorows/THREAD_NUM;
    int maxx=0;
    float32x4_t t1,t2,t3,sum;
    int choice = n % 4;
    if(id==3){
        maxx=nonzeros;

    }else{
        maxx=myindex[interval*(id+1)];
    }

    for(int i=myindex[interval*id];i<maxx;i++){//计算的index[]是从row的i行到i+interval行
            //mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
        for(int k=0;k<n-choice;k+=4)
        {
            t1=vld1q_f32(mat_nonsparse[col[i]]+k);
            sum = vdupq_n_f32(0.f);
            t3 = vdupq_n_f32(value[i]);
            t2=vld1q_f32(mat_res1[row[i]]+k);
            sum = vmulq_f32(t3,t1);
            t2=vaddq_f32(t2,sum);
            vst1q_f32(mat_res1[row[i]]+k,t2);
        }
        for(int k=n-choice;k < n;k++){
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
        }
    }
    pthread_exit(NULL);
}


///-------------------------------------------------------------实现了一堆堆线程的封装--------------------------------------------
//pThread实现spMV代码封装：静态线程分配
double spMV_pThread_static(int thread_num){
    THREAD_NUM=thread_num;
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i++)
    {
      threadParm[i].threadId = i;
      pthread_create(&thread[i], nullptr, coo_multiply_vector_pthread1, (void *)&threadParm[i]);
    }

    for (int i = 0; i < THREAD_NUM; i++)
    {
      pthread_join(thread[i], nullptr);
    }
    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

//pThread实现spMV代码封装：动态线程分配函数
double spMV_pThread_dynamic(int thread_num){
    THREAD_NUM=thread_num;
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];
    next_arr=0;
    for (int i = 0; i < THREAD_NUM; i++)
    {
      threadParm[i].threadId = i;
      pthread_create(&thread[i], nullptr, coo_multiply_vector_pthread2, (void *)&threadParm[i]);
    }

    for (int i = 0; i < THREAD_NUM; i++)
    {
      pthread_join(thread[i], nullptr);
    }
    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

//pThread实现spMM静态线程分配第一种算法封装
double spMM_pThread_static1(int thread_num){
    THREAD_NUM=thread_num;
    struct timeval val,newVal;
    gettimeofday(&val, NULL);

    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i++)
    {
      threadParm[i].threadId = i;
      pthread_create(&thread[i], nullptr, coo_multiply_matrix_pthread1, (void *)&threadParm[i]);
    }

    for (int i = 0; i < THREAD_NUM; i++)
    {
      pthread_join(thread[i], nullptr);
    }

    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

//pThread实现spMM静态线程分配第二种算法封装------------------------这种算法造成的性能开销异常大，cache水平
double spMM_pThread_static2(int thread_num){
    THREAD_NUM=thread_num;
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    pthread_t thread[THREAD_NUM];
    threadParm_t2 threadParm[THREAD_NUM];
    for (int j=0;j<nonzeros;j++)
    {
        for (int i = 0; i < THREAD_NUM; i++)
        {
            threadParm[i].threadId = i;
            threadParm[i].rowid = j;
            pthread_create(&thread[i], nullptr, coo_multiply_matrix_pthread2, (void *)&threadParm[i]);
        }
        for (int i = 0; i < THREAD_NUM; i++)
        {
            pthread_join(thread[i], nullptr);
        }
    }

    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

//pThread实现spMM代码封装：动态线程分配函数
double spMM_pThread_dynamic1(int thread_num){
    THREAD_NUM=thread_num;
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];
    next_arr2=0;
    for (int i = 0; i < THREAD_NUM; i++)
    {
      threadParm[i].threadId = i;
      pthread_create(&thread[i], nullptr, coo_multiply_matrix_pthread3, (void *)&threadParm[i]);
    }

    for (int i = 0; i < THREAD_NUM; i++)
    {
      pthread_join(thread[i], nullptr);
    }
    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

//pThread实现spMM代码封装：动态线程+neon优化
double spMM_pThread_dynamic_neon(int thread_num){
    THREAD_NUM=thread_num;
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];
    next_arr2=0;
    for (int i = 0; i < THREAD_NUM; i++)
    {
      threadParm[i].threadId = i;
      pthread_create(&thread[i], nullptr, coo_multiply_matrix_pthread4, (void *)&threadParm[i]);
    }

    for (int i = 0; i < THREAD_NUM; i++)
    {
      pthread_join(thread[i], nullptr);
    }
    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

//pThread实现spMM代码封装：静态线程+neon优化
double spMM_pThread_static_neon(int thread_num){
    THREAD_NUM=thread_num;
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];

    for (int i = 0; i < THREAD_NUM; i++)
    {
      threadParm[i].threadId = i;
      pthread_create(&thread[i], nullptr, coo_multiply_matrix_pthread_sse1, (void *)&threadParm[i]);
    }

    for (int i = 0; i < THREAD_NUM; i++)
    {
      pthread_join(thread[i], nullptr);
    }
    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}


//-----------------------------------------------------------------------------------------
//这里是要实现一堆堆的openMP的代码
///--------------------------------------------------------openMP编程中的spMV算法优化------------------------------------------
///实现spMV的openMP编程版本静态线程分配
double coo_multiply_vector_openMP_static(){
    struct timeval val,newVal;
    gettimeofday(&val, NULL);

    int i,j;
    #pragma omp parallel for num_threads(OMP_NUM_THREADS),private(i, j)
    for(i=0;i<nozerorows;i++)
    {
        for(j=myindex[i];j<myindex[i+1];j++)
        {
            yy[row[j]]+=value[j]*vec[col[j]];
        }
    }
    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

///实现spMV的openMP编程版本动态线程分配
double coo_multiply_vector_openMP_dynamic(){
    struct timeval val,newVal;
    gettimeofday(&val, NULL);

    int i,j;
    #pragma omp parallel num_threads(OMP_NUM_THREADS),private(i, j)
    //#pragma omp for schedule(static, nozerorows/OMP_NUM_THREADS)dynamic, 50
    #pragma omp for schedule(guided)
    for(i=0;i<nozerorows;i++)
    {
        for(j=myindex[i];j<myindex[i+1];j++)
        {
            yy[row[j]]+=value[j]*vec[col[j]];
        }
    }

    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

///--------------------------------------------------------openMP编程中的spMM算法优化------------------------------------------
///实现spMM的openMP编程版本静态线程分配
double coo_multiply_matrix_openMP_static(){
    struct timeval val,newVal;
    gettimeofday(&val, NULL);

    int i,k;
    #pragma omp parallel for num_threads(OMP_NUM_THREADS),private(i, k),shared(mat_res1,mat_nonsparse,row,value,col)
    for (i=0;i<nonzeros;i++)
    {
        for(k=0;k<n;k++)
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
    }

    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

///实现spMM的openMP编程版本动态线程分配
double coo_multiply_matrix_openMP_dynamic(){
    struct timeval val,newVal;
    gettimeofday(&val, NULL);

    int i,k;
    #pragma omp parallel num_threads(OMP_NUM_THREADS),private(i, k),shared(mat_res1,mat_nonsparse,row,value,col)
    //#pragma omp for schedule(static, nozerorows/OMP_NUM_THREADS)dynamic, 50,shared(mat_res1,mat_nonsparse,row,value,col)
    #pragma omp for schedule(guided)
    for (i=0;i<nonzeros;i++)
    {
        for(k=0;k<n;k++)
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
    }

    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

///实现spMM的openMP编程版本静态线程分配+NEON并行
double coo_multiply_matrix_openMP_static_neon(){
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    float32x4_t t1,t2,t3,sum;
    int choice = n % 4;
    int i,k;
    #pragma omp parallel for num_threads(OMP_NUM_THREADS),private(i, k,t1,t2,t3,sum),shared(choice,mat_nonsparse,mat_res1,value,row,col)
    for (i=0;i<nonzeros;i++)
    {
        for(int k=0;k<n-choice;k+=4)
        {
            sum=vdupq_n_f32(0.f);
            t1=vld1q_f32(mat_nonsparse[col[i]]+k);
            t2=vld1q_f32(mat_res1[row[i]]+k);
            t3 = vdupq_n_f32(value[i]);
            sum = vmulq_f32(t3,t1);
            t2= vaddq_f32(t2,sum);
            vst1q_f32(mat_res1[row[i]]+k,t2);
        }
        for(int k=n-choice;k < n;k++){
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
        }
    }

    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

///实现spMM的openMP编程版本动态线程分配+NEON并行
double coo_multiply_matrix_openMP_dynamic_neon(){
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    float32x4_t t1,t2,t3,sum;
    int choice = n % 4;
    int i,k;
    #pragma omp parallel num_threads(OMP_NUM_THREADS),private(i, k,t1,t2,t3,sum),shared(choice,mat_nonsparse,mat_res1,value,col,row)
    //#pragma omp for schedule(static, nozerorows/OMP_NUM_THREADS)dynamic, 50
    #pragma omp for schedule(guided)
    for (i=0;i<nonzeros;i++)
    {
        for(int k=0;k<n-choice;k+=4)
        {
            sum=vdupq_n_f32(0.f);
            t1=vld1q_f32(mat_nonsparse[col[i]]+k);
            t2=vld1q_f32(mat_res1[row[i]]+k);
            t3 = vdupq_n_f32(value[i]);
            sum = vmulq_f32(t3,t1);
            t2= vaddq_f32(t2,sum);
            vst1q_f32(mat_res1[row[i]]+k,t2);
        }
        for(int k=n-choice;k < n;k++){
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
        }
    }
    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}


///-------------------------------------------------------这里是一堆堆的性能测试------------------------------------------------
//spMV中所有算法的对比分析研究：
///1.对比三种算法性能：平凡算法和pThread并行算法动态加静态线程分配
///1.改变矩阵规模测试 n的范围1000——100000
///2.改变线程数目进行测试：thread_num：4——10
void spMV_all(){
    double serial,spmv1,spmv2;
    int span=1000;
    for(int i=1000;i<=10000;i+=span){
        if(i==10000)span=10000;
        init_vector();
        n=i;
        serial=coo_multiply_vector_serial(nonzeros,n,row,col,value,vec,y);
        spmv1=spMV_pThread_static(16);
        spmv2=spMV_pThread_dynamic(16);
        cout<<i<<endl
            <<serial<<endl
            <<spmv1<<endl
            <<spmv2<<endl
            <<coo_multiply_vector_openMP_static()<<endl
            <<coo_multiply_vector_openMP_dynamic()<<endl;
    }
}

///--------------------------------------------测试openMP算法的性能----------------------------------
void spMM_openMP_test(){
    init();
    n=4096;
    //测试不同线程数目对openMP程序性能的影响
    for (int k = 10; k < 100; k+=10)
    {
        cout<<"线程数目："<<k<<endl;
        OMP_NUM_THREADS=k;//openMP需要用到的线程的数目
        cout<<coo_multiply_matrix_serial()
        <<endl<<coo_multiply_matrix_openMP_static()
        <<endl<<coo_multiply_matrix_openMP_dynamic()
        <<endl<<coo_multiply_matrix_openMP_static_neon()
        <<endl<<coo_multiply_matrix_openMP_dynamic_neon();
    }
}

///检查spMM的几种算法是否运行正常，测试所有的数据：
void spMM_all_test(){
    init();
    double serial,spmm_neon,spmm_static1,spmm_static2,spmm_dynamic,spMM_static_neon,spmm_dynamic_neon,
        spmm_openmp_static,spmm_openmp_dynamic,spmm_openmp_dynamic_neon,spmm_openmp_static_neon;
    serial=coo_multiply_matrix_serial();
    spmm_neon=coo_multiply_matrix_neon();
    spmm_static1=spMM_pThread_static1(16);
    // spmm_static2=spMM_pThread_static2(4);
    spmm_dynamic=spMM_pThread_dynamic1(16);
    spMM_static_neon=spMM_pThread_static_neon(16);
    spmm_dynamic_neon=spMM_pThread_dynamic_neon(16);

    spmm_openmp_static=coo_multiply_matrix_openMP_static();
    spmm_openmp_static_neon=coo_multiply_matrix_openMP_static_neon();
    spmm_openmp_dynamic=coo_multiply_matrix_openMP_dynamic();
    spmm_openmp_dynamic_neon=coo_multiply_matrix_openMP_dynamic_neon();

    cout<<"矩阵规模 "<<n<<"  " <<"运行时间"<<endl

        <<"serial:               "<<serial<<endl
        <<"spmm_neon:            "<<spmm_neon<<endl

        <<"spmm_static:          "<<spmm_static1<<endl
        <<"spmm_openmp_static    "<<spmm_openmp_static<<endl
        //<<"spmm_static2:       "<<spmm_static2<<endl
        <<"spmm_dynamic:         "<<spmm_dynamic<<endl
        <<"spmm_openmp_dynamic   "<<spmm_openmp_dynamic<<endl

        <<"spMM_static_neon:     "<<spMM_static_neon<<endl
        <<"spmm_open_static_neon "<<spmm_openmp_static_neon<<endl
        <<"spmm_dynamic_neon:    "<<spmm_dynamic_neon<<endl
        <<"spmm_open_dyna_neon   "<<spmm_openmp_dynamic_neon<<endl;
}


///1.对比spMM几种算法的性能
///1.改变矩阵规模测试：100——10000
///2.改变线程数目：4——10
///3.改变稀疏度：0.001——0.05
///4.动态线程改变不同矩阵规模下的单个任务的单位:基本确定了矩阵进行计算的单位
void spMM_all(){
    init();
    double serial,spmm_neon,spmm_static1,spmm_static2,spmm_dynamic,spMM_static_neon,spmm_dynamic_neon,
        spmm_openmp_static,spmm_openmp_dynamic,spmm_openmp_dynamic_neon,spmm_openmp_static_neon;
    for(int n=1000;n<6000;n+=1000){

    }
}

int main()
{
    spMV_all();
    //释放内存空间
    delete []mat;
    delete []mat_nonsparse;
    delete []mat_res1;
    delete []mat_res2;
    delete []vec;
    delete []y;
    delete []yy;
    delete []value;
    delete []col;
    delete []row;
    delete []myindex;
    return 0;
}


