#include <iostream>
#include <cuda.h>

using namespace std;

__global__ void squareKernel(int *in, int *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        out[idx] = in[idx] * in[idx];
    }
}

int main() {
    const int n = 16;
    const int size = n * sizeof(int);

    int *h_in = new int[n];
    int *h_out = new int[n];

    for (int i = 0; i < n; i++) {
        h_in[i] = i;
    }

    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    squareKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    cout << "Wejscie -> Wyjscie (kwadrat):" << endl;
    for (int i = 0; i < n; i++) {
        cout << h_in[i] << " -> " << h_out[i] << endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
