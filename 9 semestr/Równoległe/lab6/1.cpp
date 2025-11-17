#include <omp.h>
#include <iostream>

using namespace std;

int main(){
    #pragma omp parallel
    {
        cout<<"tworze zadanie"<< omp_get_thread_num() << endl;
        
        for(int i=0; i<5; i++){
            #pragma omp task
            {
                cout<<"Wykonuje zadanie"<< i << omp_get_thread_num() << endl;
            }
        }
    }
}