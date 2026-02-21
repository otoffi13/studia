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

    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    
    cudaMemcpyAsync(d_in, h_in, size/2, cudaMemcpyHostToDevice, s1);
    //cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    squareKernel<<<blocksPerGrid, threadsPerBlock, 0, s1>>>(d_in, d_out, n/2);
    
    cudaMemcpyAsync(d_in + n/2, h_in +  n/2, size/2, cudaMemcpyHostToDevice, s2);

    squareKernel<<<blocksPerGrid, threadsPerBlock, 0, s2>>>(d_in + n/2, d_out + n/2, n-n/2);    

    cudaMemcpyAsync(h_out, d_out, size/2, cudaMemcpyDeviceToHost, s1);
    cudaMemcpyAsync(h_out + n/2, d_out + n/2, size/2, cudaMemcpyDeviceToHost, s2);
    //cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    cout << "Wejscie -> Wyjscie (kwadrat):" << endl;
    for (int i = 0; i < n; i++) {
        cout << h_in[i] << " -> " << h_out[i] << endl;
    }


    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
