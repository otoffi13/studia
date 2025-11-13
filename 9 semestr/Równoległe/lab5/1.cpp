#include <iostream>
#include <omp.h>

using namespace std;

void wyswietl(int* arr, int n){
    for(int i=0; i<n; i++){
        cout<<arr[i]<<" ";
    }
    cout<<endl;
}

int main(){
    int n;
    cin>>n;
    int* arr = new int[n];

    for(int i=0; i<n; i++)
        arr[i] = i;
    wyswietl(arr, n);

    int threads = n / 2;
    if(threads < 1) threads = 1;
    omp_set_num_threads(threads);

    int active = n;
    int step = 0;

    while(active > 1){
        int pairs = active / 2;
        int next_size = (active + 1) / 2;
        int* next_arr = new int[next_size];

        #pragma omp parallel for schedule(static)
        for(int i=0; i<pairs; i++){
            int idx = 2*i;
            next_arr[i] = arr[idx] + arr[idx + 1];
        }
        if(active % 2 == 1)
            next_arr[next_size - 1] = arr[active - 1];
   
        for(int i=0; i<next_size; i++)
            arr[i] = next_arr[i];

        delete[] next_arr;

        active = next_size;
        step++;
        cout<<"Krok  "<<step<<": ";
        wyswietl(arr, active);
    }

    cout<<"Wynik: "<<arr[0]<<endl;
    delete[] arr;
}