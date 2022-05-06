/**
-----------------------------------------------------------------------
本次实验是实现pThread并行的稀疏矩阵COO格式的乘法的运算，主要包含下面两个方面：
1. SpMV: 目的：1.实现并行化算法优化并与平凡算法相比较,主要分为静态并行和动态并行
2. SPMM：目的：1.实现普通并行算法优化：单纯进行循环展开分配任务；2.实现动态并行算法优化，“结束一个任务再来一个任务”；3.实现pThread与sse结合的算法

将这些数据的输出以csv的形式存储下来，多种算法与平凡算法进行对比，在不同的数据规模下

注意：本次实验要实现文件存储，另外，要注意动态内存分配的一些注意事项
------------------------------------------------------------------------
linux版
arm 鲲鹏
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
using namespace std;
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

///一些变量进行定义：全局变量
const unsigned long Converter = 1000 * 1000; // 1s == 1000 * 1000 us

typedef struct{
	int	threadId;
} threadParm_t;

typedef struct{
	int	threadId;
	int rowid;
} threadParm_t2;

int n = 4096;//矩阵规模
int THREAD_NUM=0;//线程的个数
int nonzeros=0;//稀疏矩阵中非零元素的个数
int nozerorows=0;//稀疏矩阵中不全为0的行数，这个变量是未来进行稀疏矩阵pThread算法优化的关键变量
int single_circle;//单个线程的工作量

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
int *index=NULL;

//将稀疏矩阵转化为COO表示的格式
int matrix_to_coo(float **M,int n,float* &value,int* & row,int* & col,int* & index){
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
   index=new int[nozerorows+1];
   int p=0;
   index[p++]=0;
   for(int k=1;k<a;k++){
      if(row[k]!=row[k-1]){
          index[p++]=k;
      }
   }
   index[nozerorows]=nonzeros;//这里是一个哨兵
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
    unsigned long diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

//实现COO与稠密矩阵相乘串行算法
double coo_multiply_matrix_serial(int nonzeros,int n,int* row,int* col,float* value,float**b,float**c){
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            c[i][j]=0;

    for (int i=0;i<nonzeros;i++)
        for(int k=0;k<n;k++)
            c[row[i]][k] += value[i] * b[col[i]][k];

    gettimeofday(&newVal, NULL);
    unsigned long diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

//实现COO与稠密矩阵相乘并行算法neon
///基本思路是循环展开
double coo_multiply_matrix_neon(int nonzeros,int n,int* row,int* col,float* value,float**b,float**c){
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
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
    gettimeofday(&newVal, NULL);
    unsigned long diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

//初始化
void init(){
    nonzeros=0;
    nozerorows=0;
    THREAD_NUM=4;
    single_circle=nozerorows/(THREAD_NUM*100);
    srand((int)time(0));
    double s=0.005;
    y=new float[n]{0};
    yy=new float[n]{0};
    yyy=new float[n]{0};
    generate_vector(n,vec);//生成向量
    generate_sparse_matrix(mat,n,s);//生成稀疏矩阵mat
    generate_matrix(mat_nonsparse,n);//生成稠密矩阵
    mat_res1=new float*[n];
    mat_res2=new float*[n];
    mat_res3=new float*[n];
    for(int i=0;i<n;i++)
    {
        mat_res1[i]=new float[n]{0};
        mat_res2[i]=new float[n]{0};
        mat_res3[i]=new float[n]{0};
    }
    nonzeros=matrix_to_coo(mat,n,value,row,col,index);//生成对应的COO表示的稀疏矩阵
}


//实现pThread的spmv算法1:静态线程分配
void* coo_multiply_vector_pthread1(void *parm){
    threadParm_t *p = (threadParm_t *) parm;
    int id = p->threadId;
    int seg=nozerorows/THREAD_NUM;
    for(int i=index[seg*id];i<index[seg*(id+1)];i++){
        yy[row[i]] += value[i] * vec[col[i]];
    }
    pthread_exit(nullptr);
}

int next_arr = 0;
pthread_mutex_t  mutex_task;
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
        for(int i=index[task];i<index[task+1];i++){
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
        maxx=index[interval*(id+1)];
    }

    for(int i=index[interval*id];i<maxx;i++){//计算的index[]是从row的i行到i+interval行
        for(int k=0;k<n;k++)
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
    }
    pthread_exit(NULL);
}

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

    //for(int i=index[interval*id];i<maxx;i++){//计算的index[]是从row的i行到i+interval行
    for(int k=interval*id;k<maxx;k++)
        mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
    //}
    pthread_exit(NULL);
}

int next_arr2 = 0;
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
        else maxx=index[task+single_circle];
        for(int i=index[task];i<maxx;i++){
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
        else maxx=index[task+single_circle];
        for(int i=index[task];i<maxx;i++){
            for(int k=0;k<n;k++)
                mat_res3[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
        }
    }
    pthread_exit(NULL);
}

///实现pThread编程和sse编程结合的技术，静态加neon
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
        maxx=index[interval*(id+1)];
    }

    for(int i=index[interval*id];i<maxx;i++){//计算的index[]是从row的i行到i+interval行
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
    unsigned long diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

//pThread实现spMV代码封装：动态线程分配函数
double spMV_pThread_dynamic(int thread_num){
    THREAD_NUM=thread_num;
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];

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
    unsigned long diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
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
    unsigned long diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
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
    unsigned long diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

//pThread实现spMM代码封装：动态线程分配函数
double spMM_pThread_dynamic1(int thread_num){
    THREAD_NUM=thread_num;
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];

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
    unsigned long diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

double spMM_pThread_dynamic_neon(int thread_num){
    THREAD_NUM=thread_num;
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];

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
    unsigned long diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}


//pThread实现spMM-SSE代码封装
double spMM_pThread_neon1(int thread_num){
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
    unsigned long diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}


//spMV中所有算法的对比分析研究：
///1.对比三种算法性能：平凡算法和pThread并行算法动态加静态线程分配
///1.改变矩阵规模测试 n的范围100——10000
///2.改变线程数目进行测试：thread_num：4——10
void spMV_all(){
    double serial,spmv1,spmv2;
    int span=1000;
	cout << "矩阵规模" << ' ' << "类型" <<" "<< "平均时长"<< endl;
    for(int i=1000;i<=10000;i+=span){
        n=i;
        init();
        serial=coo_multiply_vector_serial(nonzeros,n,row,col,value,vec,y);
        //int circle=1;
        /*while(serial<0.0001){
            serial+=coo_multiply_vector_serial(nonzeros,n,row,col,value,vec,y);
            circle++;
        }*/
        cout<<i<<" "<<"serial"<<" "<<serial<<endl;
        spmv1=spMV_pThread_static(4);
        spmv2=spMV_pThread_dynamic(4);
        cout<<i<<" "<<"static"<<" "<<spmv1<<endl
            <<i<<" "<<"dynamic"<<" "<<spmv2<<endl;
    }
}

///1.对比spMM几种算法的性能
///1.改变矩阵规模测试：100——10000
///2.改变线程数目：4——10
///3.改变稀疏度：0.001——0.05
///4.动态线程改变不同矩阵规模下的单个任务的单位:基本确定了矩阵进行计算的单位
void spMM_all_test(){
    init();
    double serial,spmm_neon,spmm_static1,spmm_static2,spmm_dynamic,spMM_pThread_neon,spmm_dynamic_neon;
    serial=coo_multiply_matrix_serial(nonzeros,n,row,col,value,mat_nonsparse,mat_res2);
    spmm_neon=coo_multiply_matrix_neon(nonzeros,n,row,col,value,mat_nonsparse,mat_res2);
    spmm_static1=spMM_pThread_static1(4);
    spmm_static2=spMM_pThread_static2(4);
    spmm_dynamic=spMM_pThread_dynamic1(4);
    spMM_pThread_neon=spMM_pThread_neon1(4);
    spmm_dynamic_neon=spMM_pThread_dynamic_neon(4);

    cout<<"矩阵规模 "<<n<<"  " <<"运行时间"<<endl
        <<"serial:          "<<serial<<endl
        <<"spmm_neon:        "<<spmm_neon<<endl
        <<"spmm_static1:    "<<spmm_static1<<endl
        <<"spmm_static2:    "<<spmm_static2<<endl
        <<"spmm_dynamic:    "<<spmm_dynamic<<endl
        <<"spMM_pThread_neon:"<<spMM_pThread_neon<<endl
        <<"spmm_dynamic_neon:"<<spmm_dynamic_neon<<endl;
}

void spMM_all(){
    double serial,spmm_static1,spmm_dynamic,spmm_dynamic_sse,spmm_dynamic_avx,spmm_sse,spmm_avx,spmm_pthread_sse,spmm_pthread_avx;
    int span=100;
    ofstream outFile;
	outFile.open("spmm_serial.csv", ios::out); // 打开模式可省略
	outFile << "矩阵规模" << ',' << "计算次数" << ',' << "平均时长" << endl;
	for(int i=100;i<=10000;i+=span){
        if(i==1000){
            span=1000;
        }
        n=i;
        init();
        serial=coo_multiply_matrix_serial(nonzeros,n,row,col,value,mat_nonsparse,mat_res2);
        int circle=1;
        while(serial<0.01){
            serial+=serial=coo_multiply_matrix_serial(nonzeros,n,row,col,value,mat_nonsparse,mat_res2);
            circle++;
        }
        outFile<<i<<","<<circle<<","<<serial/circle<<endl;
	}
    outFile.close();
}

///dynamic实在是太强悍了！！！！，下面进行一个测试，看看是不数据会有一个曲线变化
///实际上只是程序内部的优化机制，使得第二次执行速度异常的快
void spMM_dynamic_lab(){
    init();
    int seg=nozerorows/(THREAD_NUM*100);
    double serial,spmm_dynamic2,spmm_static1;
    serial=coo_multiply_matrix_serial(nonzeros,n,row,col,value,mat_nonsparse,mat_res2);
    cout<<"serial:          "<<serial<<endl;
    spmm_static1=spMM_pThread_static1(4);
    cout<<"spmm_static1:    "<<spmm_static1<<endl;
    single_circle=seg;
    spmm_dynamic2=spMM_pThread_dynamic1(4);
    cout<<"spmm_dynamic2:   "<<spmm_dynamic2<<endl
        <<"加速比：   "<<(double)serial/spmm_dynamic2<<"    "<<(double)spmm_static1/spmm_dynamic2<<endl;
}

int main()
{
    spMM_all_test();
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
    delete []index;
    return 0;
}


