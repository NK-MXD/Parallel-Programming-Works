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
int matrix_to_coo(float** M, int n, float*& value, int*& row, int*& col) {
    //n为矩阵行列数 nonzeros代表矩阵的非零元素个数
    int i, j;
    int a = 0;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            if (M[i][j] != 0)
                a++;
    value = new float[a];
    col = new int[a];
    row = new int[a];
    int k = 0;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (M[i][j] != 0)
            {
                row[k] = i;
                col[k] = j;
                value[k++] = M[i][j];
            }
        }
    }
    return a;
}
///2.将COO表示的稀疏矩阵转化为矩阵
void coo_to_matrix(float* value, int* col, int* row, int nonzeros, int n, float**& M) {
    M = new float* [n];
    for (int i = 0; i < n; i++)
        M[i] = new float[n];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M[i][j] = 0;
    for (int i = 0; i < nonzeros; i++)
    {
        M[row[i]][col[i]] = value[i];
    }
}

///9.实现COO与稠密矩阵相乘串行算法
double coo_multiply_matrix_serial(int nonzeros, int n, int* row, int* col, float* value, float** b, float** c) {
    double timeuse;
    gettimeofday(&time1,NULL);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            c[i][j] = 0;

    for (int i = 0; i < nonzeros; i++)
        for (int k = 0; k < n; k++)
            c[row[i]][k] += value[i] * b[col[i]][k];

    gettimeofday(&time2,NULL);
    timeuse = (time2.tv_sec - time1.tv_sec) + (double)(time2.tv_usec - time1.tv_usec)/1000000.0;
    return timeuse;
}


///10.实现COO与稠密矩阵相乘并行算法SEE1
///基本思路是将列向量相乘进行并行化处理
double coo_multiply_matrix_sse1(int nonzeros, int n, int* row, int* col, float* value, float** b, float** c) {
    __m128 t1, t3, sum;
    double timeuse;
    gettimeofday(&time1,NULL);
    int choice = nonzeros % 4;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            c[i][j] = 0;

    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i < nonzeros - choice; i += 4)
        {
            t1 = _mm_set_ps(b[col[i + 3]][k], b[col[i + 2]][k], b[col[i + 1]][k], b[col[i]][k]);
            sum = _mm_setzero_ps();
            t3 = _mm_load_ps(value + i);
            sum = _mm_mul_ps(t3, t1);
            //加了这个循环使得并行的效果被掩盖了
            for (int j = 0; j < 4; j++)
            {
                c[row[i + j]][k] += sum[j];
            }
        }
        for (int i = nonzeros - choice; i < nonzeros; i++) {
            c[row[i]][k] += value[i] * b[col[i]][k];
        }
    }
    gettimeofday(&time2,NULL);
    timeuse = (time2.tv_sec - time1.tv_sec) + (double)(time2.tv_usec - time1.tv_usec)/1000000.0;
    return timeuse;
}

///13.实现COO与稠密矩阵相乘并行算法SEE2
///基本思路是循环展开
double coo_multiply_matrix_sse2(int nonzeros, int n, int* row, int* col, float* value, float** b, float** c) {
    double timeuse;
    gettimeofday(&time1,NULL);
    __m128 t1, t2, t3, sum;
    int choice = n % 4;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            c[i][j] = 0;

    for (int i = 0; i < nonzeros; i++)
    {
        for (int k = 0; k < n - choice; k += 4)
        {
            t1 = _mm_load_ps(b[col[i]] + k);
            sum = _mm_setzero_ps();
            t3 = _mm_set_ps1(value[i]);
            t2 = _mm_load_ps(c[row[i]] + k);
            sum = _mm_mul_ps(t3, t1);
            t2 = _mm_add_ps(t2, sum);
            _mm_store_ps(c[row[i]] + k, t2);
        }
        for (int k = n - choice; k < n; k++) {
            c[row[i]][k] += value[i] * b[col[i]][k];
        }
    }
    gettimeofday(&time2,NULL);
    timeuse = (time2.tv_sec - time1.tv_sec) + (double)(time2.tv_usec - time1.tv_usec)/1000000.0;
    return timeuse;
}

///14.实现COO与稠密矩阵相乘并行算法SEE3
///这种方法的思路是先进行矩阵的分块然后进行分块矩阵的乘法的并行优化:不会了


void generate_vector(int n, float*& x) {
    x = new float[n];
    for (int i = 0; i < n; i++)
        x[i] = rand() % 10 + 1;
}

///7.生成稠密矩阵
void generate_matrix(float**& m, int n) {
    m = new float* [n];
    for (int i = 0; i < n; i++)
        m[i] = new float[n];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int x = rand() % 10;
            m[i][j] = x + 1;
        }
    }
}


///8.生成稀疏矩阵
void generate_sparse_matrix(float**& m, int n, double s) {
    //注：s为稀疏度
    m = new float* [n];
    for (int i = 0; i < n; i++)
        m[i] = new float[n];
    for (int i = 0; i < n; i++)
    {    
        for (int j = 0; j < n; j++)
        {
            int x = rand() % 100000;
            if (x > 100000 * s)
                m[i][j] = 0;
            else
                m[i][j] = x % 10 + 1;
        }
    }
}


int main()
{
    
    int n = 4096;
    srand((int)time(0));
    double s = 0.008;
    float** mat = NULL;
    float** mat_nonsparse = NULL;
    float** mat_res1 = NULL;
    float** mat_res2 = NULL;
    float** mat_res3 = NULL;
    float* value = NULL;
    int* col = NULL;
    int* row = NULL;
    float* vec = NULL;
    float* y = NULL;
    float* yy = NULL;
    float* yyy = NULL;
    generate_vector(n, vec);
    generate_vector(n, y);
    generate_vector(n, yy);
    generate_vector(n, yyy);
    generate_sparse_matrix(mat, n, s);//生成稀疏矩阵mat
    generate_matrix(mat_nonsparse, n);
    generate_matrix(mat_res1, n);
    generate_matrix(mat_res2, n);
    generate_matrix(mat_res3, n);
    int notzeros = matrix_to_coo(mat, n, value, row, col);//生成对应的COO表示的稀疏矩阵
    double serial_mul_sum = 0;
    double sse1_mul_sum = 0;
    double sse2_mul_sum = 0;
    
       
    serial_mul_sum += coo_multiply_matrix_serial(notzeros, n, row, col, value, mat_nonsparse, mat_res1);
    sse1_mul_sum += coo_multiply_matrix_sse1(notzeros, n, row, col, value, mat_nonsparse, mat_res2);
    sse2_mul_sum += coo_multiply_matrix_sse2(notzeros, n, row, col, value, mat_nonsparse, mat_res3);
    //cout << endl;
    
    cout << "矩阵规模为: " << n <<"稀疏度为："<<s << " 平均耗时:" << endl;
    cout << "平凡算法耗时:" << serial_mul_sum  << "s" << endl;
    cout << "SSE1优化耗时:" << sse1_mul_sum  << "s" << endl;
    cout << "SSE2优化耗时:" << sse2_mul_sum  << "s" << endl;

   
    return 0;

}









