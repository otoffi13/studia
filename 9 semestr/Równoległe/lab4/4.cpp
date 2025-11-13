#include <iostream>
#include <omp.h>

using namespace std;

#include <iostream>
#include <omp.h>

using namespace std;

double sredniaArytm(double a[], int n){
    if(n <= 0) return 0.0;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < n; ++i){
        sum += a[i];
    }
    return sum / n;
}

int main(){
    double arr[] = {1.0, 2.0, 3.0, 4.0};
    cout << sredniaArytm(arr, 4) << endl;
    return 0;
}
