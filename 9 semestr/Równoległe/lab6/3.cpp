#include <iostream>
#include <omp.h>

using namespace std;

long long silnia(int n, int level){
    if(n == 0) return 1;
    long long result;
    #pragma omp task shared(result) if(level < 4)
    {
        result = n * silnia(n - 1, level + 1);
    }
    #pragma omp taskwait
    return result;
}

int main(){
    #pragma omp parallel
    {
        #pragma omp single
        {
            int number = 20;
            long long result = silnia(number, 0);
            cout << "Silnia z " << number << ": " << result << endl;
        }
    }
}