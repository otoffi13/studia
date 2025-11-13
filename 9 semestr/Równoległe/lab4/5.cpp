#include <iostream>
#include <omp.h>

using namespace std;

int main(){
    const size = 20000000;
    int* arr = new int[size];

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
            for( int i=0;i<size;i++)
                for(int j=0;j<i;j++)
                    array[i] += 1;
    }


}