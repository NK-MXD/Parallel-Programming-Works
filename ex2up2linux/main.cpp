//优化算法
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
            for (int m = n; m > 1; m /= 2) // log(n)个步骤
                for (int i = 0; i < m / 2.0; i++)
                {
                    if(i==m-i-1)
                        a[i-1]+=a[i];
                    else
                        a[i]+=a[m-i-1];
                }
            sum=a[0];
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



