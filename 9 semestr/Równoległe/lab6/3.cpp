#include <iostream>
#include <omp.h>

using namespace std;

long long factorial(int n, int level){
    if(n == 0) return 1;
    long long result;
    #pragma omp task shared(result) if(level < 4)
    {
        result = n * factorial(n - 1, level + 1);
    }
    #pragma omp taskwait
    return result;
}

int main(){
    #pragma omp parallel
    {
        #pragma omp single
        {
            int number = 6;
            long long result = factorial(number, 0);
            cout << "Factorial of " << number << " is " << result << endl;
        }
    }
}