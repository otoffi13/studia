#include <iostream>
#include <cuda.h>

using namespace std;

//

__global__ void sharedKernel(int *in, int *out, int n) {
    const int radius = 2;
    __shared__ int sh[8 + 2 * radius];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x + radius;

    if (gid < n)
        sh[lid] = in[gid];

    if (threadIdx.x < radius && gid >= radius)
        sh[lid - radius] = in[gid - radius];

    if (threadIdx.x >= blockDim.x - radius && gid + radius < n)
        sh[lid + radius] = in[gid + radius];

    __syncthreads();

    if (gid < n) {
        int result = sh[lid] * sh[lid];

        if (gid - 1 >= 0 && gid + 1 < n)
            result += sh[lid - 1] * sh[lid + 1];

        if (gid - 2 >= 0 && gid + 2 < n)
            result += sh[lid - 2] * sh[lid + 2];

        out[gid] = result;
    }
}

int main() {
    int n = 32;
    int blockSize = 8;
    int h_in[n], h_out[n];
    int *d_in, *d_out;

    for (int i = 0; i < n; i++)
        h_in[i] = i + 1;

    cudaMalloc((void**)&d_in, n * sizeof(int));
    cudaMalloc((void**)&d_out, n * sizeof(int));

    cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (n + blockSize - 1) / blockSize;

    sharedKernel<<<blocks, blockSize>>>(d_in, d_out, n);

    cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);


    cout << "i | in -> out" << endl;
    for (int i = 0; i < n; i++) {
        cout << i << " | " << h_in[i] << " -> " << h_out[i] << endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
