#include <stdio.h>
#include <omp.h>

#define N 1000000
#define THRESHOLD 1000

long long sum_par(int *A, int left, int right) {
    if (right - left < THRESHOLD) {
        long long sum = 0;
        for (int i = left; i <= right; i++)
            sum += A[i];
        return sum;
    }

    int mid = (left + right) / 2;
    long long s1 = 0, s2 = 0;

    #pragma omp task shared(s1)
    s1 = sum_par(A, left, mid);

    #pragma omp task shared(s2)
    s2 = sum_par(A, mid + 1, right);

    #pragma omp taskwait
    return s1 + s2;
}

int main() {
    int *A = new int[N];
    for (int i = 0; i < N; i++)
        A[i] = 1;

    long long result = 0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            result = sum_par(A, 0, N - 1);
        }
    }

    printf("Suma = %lld\n", result);

    delete[] A;
    return 0;
}
