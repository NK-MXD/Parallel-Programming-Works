/**
-----------------------------------------------------------------------
����ʵ����ʵ��pThread���е�ϡ�����COO��ʽ�ĳ˷������㣬��Ҫ���������������棺
1. SpMV: Ŀ�ģ�1.ʵ�ֲ��л��㷨�Ż�����ƽ���㷨��Ƚ�,��Ҫ��Ϊ��̬���кͶ�̬����
2. SPMM��Ŀ�ģ�1.ʵ����ͨ�����㷨�Ż�����������ѭ��չ����������2.ʵ�ֶ�̬�����㷨�Ż���������һ����������һ�����񡱣�3.ʵ��pThread��sse��ϵ��㷨

����Щ���ݵ������csv����ʽ�洢�����������㷨��ƽ���㷨���жԱȣ��ڲ�ͬ�����ݹ�ģ��

ע�⣺����ʵ��Ҫʵ���ļ��洢�����⣬Ҫע�⶯̬�ڴ�����һЩע������
------------------------------------------------------------------------
windows�汾
codeblock
**/
#include<cstdlib>
#include<algorithm>
#include<windows.h>
#include<iostream>
#include<ctime>
#include<pthread.h>
#include <fstream>
#include <sstream>
#include<pmmintrin.h>
#include<xmmintrin.h>
#include <immintrin.h> //AVX��AVX2
using namespace std;
//��������
void generate_vector(int n,float* & x){
    x=new float[n];
    for(int i=0;i<n;i++)
        x[i]=rand()%10+1;
}

//���ɳ��ܾ���
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

//����ϡ�����
void generate_sparse_matrix(float** & m,int n,double s){
    //ע��sΪϡ���
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

///һЩ�������ж��壺ȫ�ֱ���
long long head,tail,freq;

typedef struct{
	int	threadId;
} threadParm_t;

typedef struct{
	int	threadId;
	int rowid;
} threadParm_t2;

int n = 4096;//�����ģ
int THREAD_NUM=0;//�̵߳ĸ���
int nonzeros=0;//ϡ������з���Ԫ�صĸ���
int nozerorows=0;//ϡ������в�ȫΪ0�����������������δ������ϡ�����pThread�㷨�Ż��Ĺؼ�����
int single_circle=10;//�����̵߳Ĺ�����
double s=0.005;

float **mat=NULL;//ϡ�����
float **mat_nonsparse=NULL;//���ܾ���
float **mat_res1=NULL;//�������1
float **mat_res2=NULL;//�������2
float **mat_res3=NULL;//�������2
float *vec=NULL;//����
float *y=NULL;//spmv���1
float *yy=NULL;//spmv���2
float *yyy=NULL;//spmv���3


//ϡ������ʾ������pThread����У�Ϊ�˷�����������ǽ������е��׸�Ԫ�ص��±궼�洢��index������
float *value=NULL;
int *col=NULL;
int *row=NULL;
int *index=NULL;

//��ϡ�����ת��ΪCOO��ʾ�ĸ�ʽ
int matrix_to_coo(float **M,int n,float* &value,int* & row,int* & col,int* & index){
    //nΪ���������� nonzeros�������ķ���Ԫ�ظ���
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
   index[nozerorows]=nonzeros;//������һ���ڱ�
   return a;
}

//ʵ��COO��������ˣ������㷨��
double coo_multiply_vector_serial(int nonzeros,int n,int* row,int* col,float* value,float* x,float* y){
    //����xָ�����������������ʾ����ϡ���������������
    long long head,tail,freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i=0;i<nonzeros;i++)
        y[row[i]] += value[i] * x[col[i]];//���y�еó��Ľ��������ϡ��������������˵Ľ��
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    //cout<<"serial spmv:"<<(double)(tail-head)/(double)freq<<endl;
	return (double)(tail-head)/(double)freq;
}

//ʵ��COO����ܾ�����˴����㷨
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
	//cout<<"serial:"<<(double)(tail-head)/(double)freq<<endl;
	return (double)(tail-head)/(double)freq;
}

//ʵ��COO����ܾ�����˲����㷨SEE2
//����˼·��ѭ��չ��
double coo_multiply_matrix_sse(int nonzeros,int n,int* row,int* col,float* value,float**b,float**c){
    long long head,tail,freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    __m128 t1,t2,t3,sum;
    int choice = n % 4;
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
	//cout<<"sse:"<<(double)(tail-head)/(double)freq<<endl;
	return (double)(tail-head)/(double)freq;
}

//ʵ��COO����ܾ�����˵Ĳ����㷨AVX
//ʵ��ԭ����ʵ�ְ�·����
double coo_multiply_matrix_avx(int nonzeros,int n,int* row,int* col,float* value,float**b,float**c){
    long long head,tail,freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    __m256 t1, t2, t3, sum;
    int choice = n % 8;//����
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            c[i][j] = 0;

    for (int i = 0; i < nonzeros; i++)
    {
        for (int k = 0; k < n - choice; k += 8)
        {
            sum = _mm256_setzero_ps();
            t1=_mm256_loadu_ps(b[col[i]]+k);//ע���������ѡ�ϸ�����ָ��Ļ��ᱨ��
            t3 = _mm256_set1_ps(value[i]);
            t2=_mm256_loadu_ps(c[row[i]]+k);//��ֵload������
            sum = _mm256_mul_ps(t3,t1);//��λ���
            t2=_mm256_add_ps(t2,sum);//��λ���
            _mm256_storeu_ps(c[row[i]]+k,t2);//��λ�洢
        }
        for (int k = n - choice; k < n; k++) {
            c[row[i]][k] += value[i] * b[col[i]][k];
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	return (double)(tail-head)/(double)freq;
}
//��ʼ��
void init(){
    nonzeros=0;
    nozerorows=0;
    THREAD_NUM=4;
    srand((int)time(0));
    s=0.005;
    y=new float[n]{0};
    yy=new float[n]{0};
    yyy=new float[n]{0};
    generate_vector(n,vec);//��������
    generate_sparse_matrix(mat,n,s);//����ϡ�����mat
    generate_matrix(mat_nonsparse,n);//���ɳ��ܾ���
    mat_res1=new float*[n];
    mat_res2=new float*[n];
    mat_res3=new float*[n];
    for(int i=0;i<n;i++)
    {
        mat_res1[i]=new float[n]{0};
        mat_res2[i]=new float[n]{0};
        mat_res3[i]=new float[n]{0};
    }
    nonzeros=matrix_to_coo(mat,n,value,row,col,index);//���ɶ�Ӧ��COO��ʾ��ϡ�����
    single_circle=nozerorows/(THREAD_NUM*100);
}


//ʵ��pThread��spmv�㷨1:��̬�̷߳���
void* coo_multiply_vector_pthread1(void *parm){
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    threadParm_t *p = (threadParm_t *) parm;
    int id = p->threadId;
    int seg=nozerorows/THREAD_NUM;
    for(int i=index[seg*id];i<index[seg*(id+1)];i++){
        yy[row[i]] += value[i] * vec[col[i]];
    }
    pthread_exit(nullptr);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout<<"thread "<<id<<":"<<(double)(tail-head)/(double)freq<<endl;
}

int next_arr = 0;
pthread_mutex_t  mutex_task;
//ʵ��pThread��spmv�㷨2����̬�̷߳��� ---------������ѵ��������ȵĻ��֣�4���̣߳�����ÿʮ�н��б����Ļ���
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

//ʵ��pThread��spmm�㷨
///�����߳������ֻ���ģʽ��һ����ֱ���������л��֣���һ�������ڲ���л���
///��һ��ʵ�ֵ����������л���
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

    for(int i=index[interval*id];i<maxx;i++){//�����index[]�Ǵ�row��i�е�i+interval��
        for(int k=0;k<n;k++)
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
    }
    pthread_exit(NULL);
}
///�ڶ��ַ��������ڲ���л���
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

    //for(int i=index[interval*id];i<maxx;i++){//�����index[]�Ǵ�row��i�е�i+interval��
    for(int k=interval*id;k<maxx;k++)
        mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
    //}
    pthread_exit(NULL);
}

int next_arr2 = 0;
//ʵ��pThread��̵�spmm��̬�̷߳���
void* coo_multiply_matrix_pthread4(void *parm){
    threadParm_t *p = (threadParm_t *) parm;
    int id = p->threadId;
    __m128 t1,t2,t3,sum;
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
                t1=_mm_load_ps(mat_nonsparse[col[i]]+k);
                sum = _mm_setzero_ps();
                t3 = _mm_set_ps1(value[i]);
                t2=_mm_load_ps(mat_res1[row[i]]+k);
                sum = _mm_mul_ps(t3,t1);
                t2=_mm_add_ps(t2,sum);
                _mm_store_ps(mat_res1[row[i]]+k,t2);
            }
            for(int k=n-choice;k < n;k++){
                mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
            }
        }
    }
    pthread_exit(NULL);
}

void* coo_multiply_matrix_pthread5(void *parm){
    threadParm_t *p = (threadParm_t *) parm;
    int id = p->threadId;
    __m256 t1, t2, t3, sum;
    int choice = n % 8;//����
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
            for(int k=0;k<n-choice;k+=8)
            {
                sum = _mm256_setzero_ps();
                t1=_mm256_loadu_ps(mat_nonsparse[col[i]]+k);//ע���������ѡ�ϸ�����ָ��Ļ��ᱨ��
                t3 = _mm256_set1_ps(value[i]);
                t2=_mm256_loadu_ps(mat_res1[row[i]]+k);//��ֵload������
                sum = _mm256_mul_ps(t3,t1);//��λ���
                t2=_mm256_add_ps(t2,sum);//��λ���
                _mm256_storeu_ps(mat_res1[row[i]]+k,t2);//��λ�洢
            }//����ʣ��Ԫ��
            for(int k=n-choice;k < n;k++){
                mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
            }
        }
    }
    pthread_exit(NULL);
}


//ʵ��pThread��̵�spmm��̬�̷߳���2
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

///ʵ��pThread��̺�sse��̽�ϵļ���
void* coo_multiply_matrix_pthread_sse1(void *parm){
    threadParm_t *p = (threadParm_t *) parm;
    int id = p->threadId;
    int interval=nozerorows/THREAD_NUM;
    int maxx=0;
    __m128 t1,t2,t3,sum;
    int choice = n % 4;
    if(id==3){
        maxx=nonzeros;

    }else{
        maxx=index[interval*(id+1)];
    }

    for(int i=index[interval*id];i<maxx;i++){//�����index[]�Ǵ�row��i�е�i+interval��
            //mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
        for(int k=0;k<n-choice;k+=4)
        {
            t1=_mm_load_ps(mat_nonsparse[col[i]]+k);
            sum = _mm_setzero_ps();
            t3 = _mm_set_ps1(value[i]);
            t2=_mm_load_ps(mat_res1[row[i]]+k);
            sum = _mm_mul_ps(t3,t1);
            t2=_mm_add_ps(t2,sum);
            _mm_store_ps(mat_res1[row[i]]+k,t2);
        }
        for(int k=n-choice;k < n;k++){
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
        }
    }
    pthread_exit(NULL);
}

///ʵ��pThread��̺�avx��̽�ϵļ���
void* coo_multiply_matrix_pthread_avx1(void *parm){
    threadParm_t *p = (threadParm_t *) parm;
    int id = p->threadId;
    int interval=nozerorows/THREAD_NUM;
    int maxx=0;
    __m256 t1, t2, t3, sum;
    int choice = n % 8;//����
    if(id==3){
        maxx=nonzeros;

    }else{
        maxx=index[interval*(id+1)];
    }

    for(int i=index[interval*id];i<maxx;i++){
        for(int k=0;k<n-choice;k+=8)
        {
            sum = _mm256_setzero_ps();
            t1=_mm256_loadu_ps(mat_nonsparse[col[i]]+k);//ע���������ѡ�ϸ�����ָ��Ļ��ᱨ��
            t3 = _mm256_set1_ps(value[i]);
            t2=_mm256_loadu_ps(mat_res1[row[i]]+k);//��ֵload������
            sum = _mm256_mul_ps(t3,t1);//��λ���
            t2=_mm256_add_ps(t2,sum);//��λ���
            _mm256_storeu_ps(mat_res1[row[i]]+k,t2);//��λ�洢
        }//����ʣ��Ԫ��
        for(int k=n-choice;k < n;k++){
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
        }
    }
    pthread_exit(NULL);
}

///ʵ��pThread��̺�avx��̽�ϵļ���
/*void* coo_multiply_matrix_pthread_avx2(void *parm){
    threadParm_t *p = (threadParm_t *) parm;
    int id = p->threadId;
    int interval=nozerorows/THREAD_NUM;
    int maxx=0;
    __m512 t1, t2, t3,sum;
    int choice = n % 16;//����
    if(id==3){
        maxx=nonzeros;

    }else{
        maxx=index[interval*(id+1)];
    }

    for(int i=index[interval*id];i<maxx;i++){
        for(int k=0;k<n-choice;k+=16)
        {
            sum = _mm512_setzero_ps();
            t1=_mm512_loadu_ps(mat_nonsparse[col[i]]+k);//ע���������ѡ�ϸ�����ָ��Ļ��ᱨ��
            t3 = _mm512_set1_ps(value[i]);
            t2=_mm512_loadu_ps(mat_res1[row[i]]+k);//��ֵload������
            sum = _mm512_mul_ps(t3,t1);//��λ���
            t2=_mm512_add_ps(t2,sum);//��λ���
            _mm512_storeu_ps(mat_res1[row[i]]+k,t2);//��λ�洢
            cout<<i<<" "<<k<<endl;
        }//����ʣ��Ԫ��
        for(int k=n-choice;k < n;k++){
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
        }
    }
    pthread_exit(NULL);
}*/

//pThreadʵ��spMV�����װ����̬�̷߳���
double spMV_pThread_static(int thread_num){
    THREAD_NUM=thread_num;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
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
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout<<"pThread:"<<(double)(tail-head)/(double)freq<<endl;
	return (double)(tail-head)/(double)freq;
}

//pThreadʵ��spMV�����װ����̬�̷߳��亯��
double spMV_pThread_dynamic(int thread_num){
    THREAD_NUM=thread_num;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
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
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	//cout<<"pThread:"<<(double)(tail-head)/(double)freq<<endl;
	return (double)(tail-head)/(double)freq;
}

//pThreadʵ��spMM��̬�̷߳����һ���㷨��װ
double spMM_pThread_static1(int thread_num){
    THREAD_NUM=thread_num;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

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

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	return (double)(tail-head)/(double)freq;
}

//pThreadʵ��spMM��̬�̷߳���ڶ����㷨��װ------------------------�����㷨��ɵ����ܿ����쳣��cacheˮƽ
double spMM_pThread_static2(int thread_num){
    THREAD_NUM=thread_num;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
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

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	return (double)(tail-head)/(double)freq;
}

//pThreadʵ��spMM�����װ����̬�̷߳��亯��
double spMM_pThread_dynamic1(int thread_num){
    THREAD_NUM=thread_num;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
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
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	//cout<<"pThread:"<<(double)(tail-head)/(double)freq<<endl;
	return (double)(tail-head)/(double)freq;
}

double spMM_pThread_dynamic_sse(int thread_num){
    THREAD_NUM=thread_num;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
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
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	//cout<<"pThread:"<<(double)(tail-head)/(double)freq<<endl;
	return (double)(tail-head)/(double)freq;
}

double spMM_pThread_dynamic_avx(int thread_num){
    THREAD_NUM=thread_num;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];

    for (int i = 0; i < THREAD_NUM; i++)
    {
      threadParm[i].threadId = i;
      pthread_create(&thread[i], nullptr, coo_multiply_matrix_pthread5, (void *)&threadParm[i]);
    }

    for (int i = 0; i < THREAD_NUM; i++)
    {
      pthread_join(thread[i], nullptr);
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	//cout<<"pThread:"<<(double)(tail-head)/(double)freq<<endl;
	return (double)(tail-head)/(double)freq;
}

//pThreadʵ��spMM-SSE�����װ
double spMM_pThread_sse1(int thread_num){
    THREAD_NUM=thread_num;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
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
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	//cout<<"pThread:"<<(double)(tail-head)/(double)freq<<endl;
	return (double)(tail-head)/(double)freq;
}

//pThreadʵ��spMM-AVX�����װ
double spMM_pThread_avx1(int thread_num){
    THREAD_NUM=thread_num;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];

    for (int i = 0; i < THREAD_NUM; i++)
    {
      threadParm[i].threadId = i;
      pthread_create(&thread[i], nullptr, coo_multiply_matrix_pthread_avx1, (void *)&threadParm[i]);
    }

    for (int i = 0; i < THREAD_NUM; i++)
    {
      pthread_join(thread[i], nullptr);
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	//cout<<"pThread:"<<(double)(tail-head)/(double)freq<<endl;
	return (double)(tail-head)/(double)freq;
}

//pThreadʵ��spMM-AVX�����װ2
/*double spMM_pThread_avx2(int thread_num){
    THREAD_NUM=thread_num;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];

    for (int i = 0; i < THREAD_NUM; i++)
    {
      threadParm[i].threadId = i;
      pthread_create(&thread[i], nullptr, coo_multiply_matrix_pthread_avx2, (void *)&threadParm[i]);
    }

    for (int i = 0; i < THREAD_NUM; i++)
    {
      pthread_join(thread[i], nullptr);
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	//cout<<"pThread:"<<(double)(tail-head)/(double)freq<<endl;
	return (double)(tail-head)/(double)freq;
}*/


//spMV�������㷨�ĶԱȷ����о���
///1.�Ա������㷨���ܣ�ƽ���㷨��pThread�����㷨��̬�Ӿ�̬�̷߳���
///1.�ı�����ģ���� n�ķ�Χ100����10000
///2.�ı��߳���Ŀ���в��ԣ�thread_num��4����10
void spMV_all(){
    double serial,spmv1,spmv2;
    int span=100;
    /*ofstream outFile;
	outFile.open("spmv_serial.csv", ios::out); // ��ģʽ��ʡ��
	outFile << "�����ģ" << ',' << "�������" << ',' << "ƽ��ʱ��" << endl;*/

	ofstream outFile;
	outFile.open("spmv_pthread_static.csv", ios::out); // ��ģʽ��ʡ��
	outFile << "�����ģ" << ',' << "�߳���Ŀ" << ',' << "�������" <<',' << "ƽ��ʱ��"<< endl;
    for(int i=100;i<=10000;i+=span){
        if(i==1000){
            span=1000;
        }
        n=i;
        init();
        //serial=coo_multiply_vector_serial(nonzeros,n,row,col,value,vec,y);
        //int circle=1;
        /*while(serial<0.0001){
            serial+=coo_multiply_vector_serial(nonzeros,n,row,col,value,vec,y);
            circle++;
        }*/
        //outFile<<i<<","<<circle<<","<<serial/circle<<endl;

        for(int j=4;j<=12;j++){
            /*spmv2=spMV_pThread_dynamic(j);
            int circle=1;
            if(spmv2<0.0001){
                spmv2+=spMV_pThread_dynamic(j);
                circle++;
            }
            outFile<<i<<","<<j<<","<<circle<<","<<spmv2/circle<<endl;*/
            spmv1=spMV_pThread_static(j);
            int circle=1;
            if(spmv1<0.0001){
                spmv1+=spMV_pThread_static(j);
                circle++;
            }
            outFile<<i<<","<<j<<","<<circle<<","<<spmv1/circle<<endl;
        }
    }
    outFile.close();
}

///1.�Ա�spMM�����㷨������
///1.�ı�����ģ���ԣ�100����10000
///2.�ı��߳���Ŀ��4����10
///3.�ı�ϡ��ȣ�0.001����0.05
///4.��̬�̸߳ı䲻ͬ�����ģ�µĵ�������ĵ�λ:����ȷ���˾�����м���ĵ�λ
void spMM_all_test(){
    init();
    double serial,spmm_static1,spmm_dynamic,spmm_dynamic_sse,spmm_dynamic_avx,spmm_sse,spmm_avx,spmm_pthread_sse,spmm_pthread_avx;
    serial=coo_multiply_matrix_serial(nonzeros,n,row,col,value,mat_nonsparse,mat_res2);
    spmm_sse=coo_multiply_matrix_sse(nonzeros,n,row,col,value,mat_nonsparse,mat_res2);
    spmm_avx=coo_multiply_matrix_avx(nonzeros,n,row,col,value,mat_nonsparse,mat_res1);
    spmm_static1=spMM_pThread_static1(4);
    spmm_dynamic=spMM_pThread_dynamic1(4);
    spmm_pthread_sse=spMM_pThread_sse1(4);
    //spmm_dynamic_sse=spMM_pThread_dynamic_sse(4);
    spmm_pthread_avx=spMM_pThread_avx1(4);
    //spmm_dynamic_avx=spMM_pThread_dynamic_avx(4);

    cout<<"�����ģ "<<n<<"  " <<"����ʱ��"<<endl
        <<"serial:          "<<serial<<endl
        <<"spmm_sse:        "<<spmm_sse<<endl
        <<"spmm_avx:        "<<spmm_avx<<endl
        <<"spmm_static1:    "<<spmm_static1<<endl
        <<"spmm_dynamic:    "<<spmm_dynamic<<endl
        <<"spmm_dynamic_sse:"<<spmm_dynamic_sse<<endl
        <<"spmm_dynamic_avx:"<<spmm_dynamic_avx<<endl
        <<"spmm_pthread_sse:"<<spmm_pthread_sse<<endl
        <<"spmm_pthread_avx:"<<spmm_pthread_avx<<endl;
}

void spMM_all(){
    double serial,spmm_static1,spmm_dynamic,spmm_dynamic_sse,spmm_dynamic_avx,spmm_sse,spmm_avx,spmm_pthread_sse,spmm_pthread_avx;
    int span1=100;
    double span2=0.001;
	ofstream outFile;
	outFile.open("spmm_pthread_all.csv", ios::out); // ��ģʽ��ʡ��
	outFile << "�����ģ*ϡ���" << ',' << "ƽ���㷨ʱ��"<< ',' << "SSEʱ��"<< ','
        << "AVXƽ��ʱ��"<< ',' << "��̬�̷߳���ʱ��"<< ',' << "��̬�̷߳���ʱ��"
        << ',' << "��̬�߳�+sseʱ��"<< ',' << "��̬�߳�+avxʱ��"<< endl;
	for(int i=1000;i<=10000;i+=span1){
        if(i==1000){
            span1=1000;
        }
        for(double j=0.001;j<0.05;j+=span2){
            if(j==0.01)span2=0.01;
            n=i;
            s=j;
            init();
            serial=coo_multiply_matrix_serial(nonzeros,n,row,col,value,mat_nonsparse,mat_res2);
            spmm_sse=coo_multiply_matrix_sse(nonzeros,n,row,col,value,mat_nonsparse,mat_res2);
            spmm_avx=coo_multiply_matrix_avx(nonzeros,n,row,col,value,mat_nonsparse,mat_res1);
            spmm_static1=spMM_pThread_static1(4);
            spmm_dynamic=spMM_pThread_dynamic1(4);
            spmm_pthread_sse=spMM_pThread_sse1(4);
            spmm_pthread_avx=spMM_pThread_avx1(4);
            outFile<<nonzeros<<","<<serial<<","<<spmm_sse<<","
                <<spmm_avx<<","<<spmm_static1<<","<<spmm_dynamic<<","
                <<spmm_pthread_sse<<","<<spmm_pthread_avx<<endl;
        }

        /*serial=coo_multiply_matrix_serial(nonzeros,n,row,col,value,mat_nonsparse,mat_res2);
        int circle=1;
        while(serial<0.0001){
            serial+=serial=coo_multiply_matrix_serial(nonzeros,n,row,col,value,mat_nonsparse,mat_res2);
            circle++;
        }
        outFile<<i<<","<<circle<<","<<serial/circle<<endl;*/
        /*for(int j=4;j<=12;j++){
            /*spmv2=spMV_pThread_dynamic(j);
            int circle=1;
            if(spmv2<0.0001){
                spmv2+=spMV_pThread_dynamic(j);
                circle++;
            }
            outFile<<i<<","<<j<<","<<circle<<","<<spmv2/circle<<endl;
            spmv1=spMV_pThread_static(j);
            cout<<i<<","<<j<<","<<circle<<","<<spmv1/circle<<endl;
        }*/
	}
    outFile.close();
}

///dynamicʵ����̫ǿ���ˣ����������������һ�����ԣ������ǲ����ݻ���һ�����߱仯
///ʵ����ֻ�ǳ����ڲ����Ż����ƣ�ʹ�õڶ���ִ���ٶ��쳣�Ŀ�
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
        <<"���ٱȣ�   "<<(double)serial/spmm_dynamic2<<"    "<<(double)spmm_static1/spmm_dynamic2<<endl;
}

int main()
{
    n=4096;
    init();
    spMV_pThread_static(4);
    //�ͷ��ڴ�ռ�
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


