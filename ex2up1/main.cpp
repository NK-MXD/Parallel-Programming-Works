//”≈ªØÀ„∑®
#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

double a[10000],sum,sum1,sum2;

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
            for (int i = 0;i < n; i += 2) {
                sum1 += a[i];
                sum2 += a[i + 1];
            }
            sum = sum1 + sum2;
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
