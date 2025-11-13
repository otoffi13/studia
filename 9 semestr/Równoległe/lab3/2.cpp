#include <iostream>
#include <ctime>
#include <omp.h>


inline double random_double(unsigned int &seed) {
    seed = 1664525 * seed + 1013904223;
    return (double)(seed & 0x00FFFFFF) / 0x01000000;
}

double monteCarloPi(long long N) {
    long long count = 0;

    #pragma omp parallel
    {
        unsigned int seed = (unsigned int)(time(NULL)) ^ (omp_get_thread_num() * 7919);
        long long local_count = 0;

        #pragma omp for
        for (long long i = 0; i < N; i++) {
            double x = random_double(seed);
            double y = random_double(seed);
            if (x * x + y * y <= 1.0)
                local_count++;
        }

        #pragma omp atomic
        count += local_count;
    }

    return 4.0 * (double)count / (double)N;
}

int main() {
    long long N = 100000;
    double pi = monteCarloPi(N);
    std::cout << pi << std::endl;
    return 0;
}
