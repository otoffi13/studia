#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <random>

using namespace std;

int main(){
    int a;
    cin>>a;

    int* arr1 = new int[a];
    int* arr2 = new int[a];

    #pragma omp parallel
    {

        std::mt19937 gen(static_cast<unsigned int>(time(nullptr)));
        std::uniform_int_distribution<int> dist(0, 9);

        #pragma omp for schedule(static,5)
        for(int i = 0; i < a; ++i){
            arr1[i] = i;
            arr2[i] = dist(gen);
        }

        #pragma omp for schedule(static,5)
        for(int i = 0; i < a; ++i){
            arr1[i] = arr1[i] + arr2[i];
        }
    }

    for(int i=0;i<a;i++)
        cout<<arr1[i]<<" "<<arr2[i]<<endl;

    delete[] arr1;
    delete[] arr2;
    return 0;
}