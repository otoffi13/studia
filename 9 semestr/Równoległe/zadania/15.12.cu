#include <iostream>
#include <cuda.h>
#include <bits/stdc++.h>

using namespace std;

__global__ void sumKernel(int *in, int n, int half) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx + half < n){
        in[idx] += in[idx + half];
    }
}

__global__ void kernel(int *in, int n, int idx){
    in[0] += in[idx];
}

int main() {
    const int n = 16;
    const int size = n * sizeof(int);

    int *h_in = new int[n];
    int blockSize = 1024;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    for (int i = 0; i < n; i++) {
        h_in[i] = i;
    }

    int *d_in;
    cudaMalloc((void**)&d_in, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    int half = n;
    while ( half > 1){
        if (half % 2 == 1){
            kernel<<<1,1>>>(d_in, n, half - 1);
        }
        half = half / 2;
        sumKernel<<<gridSize, blockSize>>>(d_in, n, half);
    }
    cudaMemcpy(h_in, d_in, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);

    cout << "Suma elementow tablicy: " << h_in[0] << endl;
    return 0;
}
