#include <iostream>

using namespace std;

// Kernel CUDA
__global__ void addKernel(int *c, const int *a, const int *b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(){
    const int arraySize = 5;
    const int bytes = arraySize * sizeof(int);
    
    // Dane na CPU
    int h_a[arraySize] = {1, 2, 3, 4, 5};
    int h_b[arraySize] = {10, 20, 30, 40, 50};
    int h_c[arraySize] = {0};
    
    // Alokacja pamięci GPU
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Kopiowanie danych CPU -> GPU
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Uruchomienie kernela
    addKernel<<<1, arraySize>>>(d_c, d_a, d_b);
    
    // Kopiowanie wyników GPU -> CPU
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Wyświetlenie rezultatów
    cout << "Wynik dodawania:\n";
    for (int i = 0; i < arraySize; i++) {
        cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << endl;
    }
    
    // Zwolnienie pamięci GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}