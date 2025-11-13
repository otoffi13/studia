#include <iostream>
#include <stdio.h>
using namespace std;

#define uint unsigned int

class Gora{
    public:
        uint id;
        uint wysokosc;
        string nazwa;

};

bool czyPotegaDwojki(uint a){
    uint pom=a&(a-1);
    if(a==0 || pom==0)
        return true;
    else return false;
}

void sortowanie(Gora *tab, int poczatek, int koniec){
    int i, j;
    Gora key;
    for(i=poczatek+1;i<koniec;++i){
        key=tab[i];
        j=i-1;
        while(j>=poczatek && tab[j].wysokosc>key.wysokosc){
            tab[j+1]=tab[j];
            j=j-1;
        }
        tab[j+1]=key;
    }
    for(int i=poczatek;i<koniec-1;++i){
        if(tab[i].wysokosc==tab[i+1].wysokosc && tab[i].id>tab[i+1].id)
            swap(tab[i], tab[i+1]);
    }
}

int main(){
        std::ios_base::sync_with_stdio(false);
        std::cout.tie(nullptr);
        std::cin.tie(nullptr);

        uint t, n;                     //t-liczba testów, n-liczba szczytów
        Gora *tab;
        char tmp[25];

        scanf("%u", &t);
        for(int i=1;i<=t;++i){
            scanf("%u", &n);
            tab=new Gora[n];
            for(int j=0;j<n;++j){
                tab[j].id=j;
                scanf("%s %u", &tmp, &tab[j].wysokosc);
                tab[j].nazwa=tmp;
            }

            int liczbaPoteg=0;
            for(int j=0;j<n;++j){
                if(czyPotegaDwojki(tab[j].wysokosc)){
                    swap(tab[j], tab[liczbaPoteg]);
                    ++liczbaPoteg;
                }
            }
           sortowanie(tab, 0, liczbaPoteg);
           sortowanie(tab, liczbaPoteg, n);
            for(int k=0;k<n;k++)
                cout<<tab[k].nazwa<<"-"<<tab[k].wysokosc<<" ";
            cout<<endl;
            delete[] tab;

        }
    return 0;
}