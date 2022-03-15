//”≈ªØÀ„∑®
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>

using namespace std;

double a[10000],sum,sum1,sum2;

int main()
{
    clock_t start,finish;

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
        start = clock ();
        while (clock()-start < 10) {
            counter++;
            for (int i = 0;i < n; i += 2) {
                sum1 += a[i];
                sum2 += a[i + 1];
            }
            sum = sum1 + sum2;
            finish = clock ();
        }
        msseconds = (finish-start)*1000/float (CLOCKS_PER_SEC);
        //for(int i=0;i<n;i++){
        //    cout<<sum[i]<<" ";
        //}
        cout <<n <<' '<< counter <<' '<< msseconds<<' '<<fixed <<setprecision(5)<< msseconds / counter << endl ;
        if (n == 100) step = 100;
        if(n==1000)step=1000;
    }
    return 0;
}



