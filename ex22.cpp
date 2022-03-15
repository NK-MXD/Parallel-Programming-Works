#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>

using namespace std;
double a[10000],sum;
double normal(int n,double a[])
{
    double sum=0.0;
    for(int i = 0; i < n; i++)
        sum+=a[i];
    return sum;
}
double improve1(int n,double a[])
{
    double sum1=0,sum2=0;
    double sum=0.0;
    for (int i = 0;i < n; i += 2) {
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    sum = sum1 + sum2;
    return sum;
}
double improve2(int n,double a[])
{
    double sum=0.0;
    for (int m = n; m > 1; m /= 2)
        for (int i = 0; i < m / 2.0; i++)
        {
            if(i==m-i-1)
                a[i-1]+=a[i];
            else
                a[i]+=a[m-i-1];
        }
    sum=a[0];
    return sum;
}
int main()
{
    clock_t start,finish;

    int n, step = 10;
    long counter ;
    double msseconds ;
    for (n = 0; n <= 10000; n += step) {
        for (int i = 0; i < n; i++)
        {
            sum=0.0;
            a[i]=i;
        }
        counter = 0;
        start = clock ();
        while (clock()-start < 10) {
            counter++;
            sum=normal(n,a);
            finish = clock ();
        }
        msseconds = (finish-start)*1000/float (CLOCKS_PER_SEC);
        cout <<"normal:"<<n <<' '<< counter <<' '<< msseconds<<' '<<fixed <<setprecision(5)<< msseconds / counter << endl ;
        start = clock ();
        while (clock()-start < 10) {
            counter++;
            sum=improve1(n,a);
            finish = clock ();
        }
        msseconds = (finish-start)*1000/float (CLOCKS_PER_SEC);
        cout <<"improve1:"<<n <<' '<< counter <<' '<< msseconds<<' '<<fixed <<setprecision(5)<< msseconds / counter << endl ;
        for (int i = 0; i < n; i++)
        {
            sum=0.0;
            a[i]=i;
        }
        start = clock ();
        while (clock()-start < 10) {
            counter++;
            for(int i=0;i<n;i++)
                a[i]=i;
            sum=improve2(n,a);
            finish = clock ();
        }
        msseconds = (finish-start)*1000/float (CLOCKS_PER_SEC);
        cout <<"improve2:"<<n <<' '<< counter <<' '<< msseconds<<' '<<fixed <<setprecision(5)<< msseconds / counter << endl ;
        if (n == 100) step = 100;
        if(n==1000)step=1000;
    }
    return 0;
}
