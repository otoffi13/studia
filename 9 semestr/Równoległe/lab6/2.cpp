#include <iostream>
#include <omp.h>

using namespace std;

int fib(int index, int level){
    if(index == 0) return 0;
    if(index == 1) return 1;
    int prev1, prev2;
    #pragma omp task shared(prev1) if(level < 4)
    {
        prev1 = fib(index - 1, level + 1);
    }
    #pragma omp task shared(prev2) if(level < 4)
    {
        prev2 = fib(index - 2, level + 1);
    }
    #pragma omp taskwait
        return prev1 + prev2;
}

int main(){
    #pragma omp parallel
    {
        #pragma omp single
        {
            int wynik = fib(41, 0);
            cout<<"Wynik: " << wynik << endl;
        }
    }
}