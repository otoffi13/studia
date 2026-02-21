#include <iostream>
#include <cuda.h>
#include <bits/stdc++.h>

using namespace std;

__global__ void squareKernel(int *in, int *out, int n) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; 
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

    dim3 threadsPerBlock(sqrt(n)+1, sqrt(n)+1);
    dim3 blocksPerGrid((sqrt(n) + threadsPerBlock.x - 1) / threadsPerBlock.x, (sqrt(n) + threadsPerBlock.y - 1) / threadsPerBlock.y);

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
