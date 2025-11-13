
#include <iostream>
#include <stdio.h>
#include <cmath>

using namespace std;


    //------zad1
    //float a=.1, b=0.9, c=-1;
    //float res1 = (a+b)+c;
    //float res2 = a+(b+c);
    //printf("%.10f\t%.10f\n", res1, res2);

    //------zad 2
    void gen(float *arr, int n, int m){
        for(int i=0; i<n;i++){
            int ii=i%m;
            arr[i]=1./((ii+1)*ii+2);
        }
    }
    float sum(float *arr, int n){
        float sum=0;
        for (int i=0;i<n;i++){
            sum+=arr[i];
        }
        return sum;
    }
    float sumExact(int n, int m){
        return (float)n/(m+1);
    }

    int main(){
        int n=1<<20;
        int m=1<<5;
        float *arr=malloc(n*sizeof(float));
        gen(arr, n, m);
        float sumOrd=sum(arr, n);
        float exact=sumExact(n, m);
        printf("%.10f = Sumowanie zwykłe \n%.10f = Dokładna", sumOrd, exact);
        float sumGM=

        printf("%e = błąd względny\n", fabs((sumGM-exact)/exact);
        printf("%e = blad w")
    }
