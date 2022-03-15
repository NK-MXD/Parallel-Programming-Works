#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

double a[10000],b[10000][10000],sum[10000];

int main()
{
    long long head, tail, freq;        // timers
    clock_t start,finish;
    int n, step = 10;
    long counter ;
    double msseconds ;
    for (n = 0; n <= 10000; n += step) {
        for (int i = 0; i < n; i++)
        {
            a[i]=1;
            sum[i]=0.0;
            for (int j = 0; j < n; j++)
                b[i][j] = i + j;
        }
        counter = 0;
        start = clock ();
        while (clock()-start < 10) {
            counter++;
            for(int i=0;i<n;i++)
                sum[i]=0.0;
            for(int j = 0; j < n;j++)
            {
                for(int i = 0;i < n;i++)
					sum[i]+=b[j][i]*a[j];
            }
            finish = clock ();
        }
        msseconds = (finish-start)/float (CLOCKS_PER_SEC);
        //for(int i=0;i<n;i++){
        //    cout<<sum[i]<<" ";
        //}
        cout <<n <<' '<< counter <<' '<< msseconds<<' '<< msseconds / counter << endl ;
        if (n == 100) step = 100;
        if(n==1000)step=1000;
    }
    return 0;
}
