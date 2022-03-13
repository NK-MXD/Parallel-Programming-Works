//改进思路二
#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

double a[10000],sum,sum1,sum2;

double  recursion(int n)
{
    if (n == 1)
        return a[0];
    else
    {
        for(int i = 0; i < n / 2; i++)
            a[i]+=a[n-i-1];
        n = n / 2;
        recursion(n);
    }
}

int main()
{
    long long head, tail, freq;        // timers

	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);	// similar to CLOCKS_PER_SEC

    int n, step = 10;
    long counter ;
    double msseconds ;
    for (n = 0; n <= 10000; n += step) {
        for (int i = 0; i < n; i++)
        {
            sum1 = 0;
            sum2 = 0;
            sum=0;
            a[i]=i;
        }
        counter = 0;
        QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        while ((tail - head) * 1000.0 / freq < 1) {
            counter++;
            for (int m = n; m > 1; m /= 2) // log(n)个步骤
                for (int i = 0; i < m / 2.0; i++)
                {
                    if(i==m-i-1)
                        a[i-1]+=a[i];
                    else
                        a[i]+=a[m-i-1];
                }
            sum=a[0];
            QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
        }
        msseconds = (tail - head) * 1000.0 / freq;
        //for(int i=0;i<n;i++){
        //    cout<<sum[i]<<" ";
        //}
        cout <<n <<' '<< counter <<' '<< msseconds<<' '<< msseconds / counter << endl ;
        if (n == 100) step = 100;
        if(n==1000)step=1000;
    }
    return 0;
}
