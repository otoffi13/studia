#include <iostream>

using namespace std;
using uint=unsigned int;

struct Wojownik {
string name;
int crabs;
int traps;
};

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cout.tie(nullptr);
    std::cin.tie(nullptr);

    int x;
    cin >> x;
    Wojownik w;
    string tab[100][10]={};
    for(int i=0;i<x;++i){
        cin>>w.name>>w.crabs>>w.traps;
        if(tab[w.crabs][w.traps]!="")
            tab[w.crabs][w.traps]+=" "+w.name;
        else tab[w.crabs][w.traps]=w.name;
        }
    for(int j=99;j>=0;--j){
        for(int k=0;k<=9;++k)
                if(tab[j][k]!="")
                    cout<<tab[j][k]<<" ";
    }
return 0;
}