#include <iostream>
#include <omp.h>

using namespace std;


int main(){
    const int size = 10000;
    int array[size];

    //jak zrównoleglić tą pętle

    // for( int i = 0;i<size;i++){
    //     array[i] = i;
    // }
    // for(int i = 0; i<size;i++){
    //     array[i] *= array[i];
    // }
    for( int i = 0;i<size;i++){
            array[i] = i;
        }   
    #pragma omp parallel
    {
    int index = omp_get_thread_num();
    int threads = omp_get_num_threads();

    for(int i =index;i<size;i+=threads)
        array[i] *= array[i];
    
    }
}