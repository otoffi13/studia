#include <iostream>

using namespace std;

int main(){
    const int size = 1000000;
    int* arrayhost = new int[size];

    for(int i = 0; i < size; i++){
        arrayhost[i] = i;
    }
    //-----------GPU CODE HERE----------------
    int* arrayGpu;
    cudaMalloc(&arrayGpu, size * sizeof(int));
    cudaMemcpy(arrayGpu, arrayhost, size * sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<(size + 255) / 256, 256>>>(arrayGpu, size);

    cudaMemcpy(arrayhost, arrayGpu, size * sizeof(int), cudaMemcpyDeviceToHost);
    //----------------------------------------

    cudaFree(arrayGpu);

}