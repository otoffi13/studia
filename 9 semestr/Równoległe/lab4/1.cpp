#include <iostream>
#include <omp.h>

using namespace std;

int main(){
    const int size = 20000000;
    int* arr = new int[size];


    #pragma omp parallel
    {
        #pragma omp for
        for(int i=0;i<size;i++){
            arr[i]=i;
        }
    }
    return 0;
}