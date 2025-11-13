#include <iostream>
#include <omp.h>

using namespace std;

int main(){
    int size = 2000000;
    int* arr = new int[size];

    #pragma omp parallel for
    for(int i=0;i<size;i++)
        arr[i] = i%10;
    
    int sum = 0;
    #pragma omp parallel for reduction(+: sum)
    for(int i=0;i<size;i++)
        sum+=arr[i];

    cout<<sum<<endl;
}