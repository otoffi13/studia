#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

#define uint unsigned int

uint n;
uint** szachownica;
uint* pozycja;
bool* wybrane;
uint najmniejszyKoszt = 99999999;
uint* wynik;

void znajdzNajlepszeHetmany(uint wiersz, uint koszt) {\

    if (wiersz == n) {
        if (koszt < najmniejszyKoszt) {
            najmniejszyKoszt = koszt;
            for(uint k=0;k<n;k++)
                wynik[k]=pozycja[k];
        }
        return;
    }

    for (uint kolumna = 0; kolumna < n; kolumna++) {
            if(najmniejszyKoszt<koszt)
                break;
        if (!wybrane[kolumna]) {
            bool czySzachowane = false;
            for (uint i = 0; i < wiersz; i++) {
                if (pozycja[i] == kolumna ||
                    pozycja[i] - i == kolumna - wiersz ||
                    pozycja[i] + i == kolumna + wiersz) {
                    czySzachowane = true;
                    break;
                }
            }
            if (!czySzachowane) {
                pozycja[wiersz] = kolumna;
                wybrane[kolumna] = true;
                znajdzNajlepszeHetmany(wiersz + 1, koszt + szachownica[wiersz][kolumna]);
                wybrane[kolumna] = false;
            }
        }
    }
}

int main() {

    std::ios_base::sync_with_stdio(false);
    std::cout.tie(nullptr);
    std::cin.tie(nullptr);
    cin >> n;

    szachownica = new uint*[n];
    pozycja = new uint[n];
    wybrane = new bool[n];
    wynik=new uint[n];

    for (uint i = 0; i < n; i++) {
        szachownica[i] = new uint[n];
        for (uint j = 0; j < n; j++) {
            cin >> szachownica[i][j];
        }
    }

    znajdzNajlepszeHetmany(0, 0);

    for (uint i = 0; i < n; i++) {
        cout << wynik[i]<<" ";
    }
    cout << endl;

    delete[] szachownica;
    delete[] pozycja;
    delete[] wybrane;
}