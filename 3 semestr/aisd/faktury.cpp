#include <iostream>

using namespace std;

#define uint unsigned int

struct node{
    struct node *left, *right, *top;
    string name;
    uint counter=1;
};

struct node *root=NULL;

void add_node(string a){
    struct node *now=root;
    struct node *addedNode=new node;

    addedNode->left=NULL;
    addedNode->right=NULL;
    if(root==NULL){
        addedNode->top=NULL;
        addedNode->name=a;
        root=addedNode;
        return;
    }
    else{
        while(now!=NULL){
            if (now->name < a) {
                if (now->right==NULL) {
                    addedNode->name=a;
                    addedNode->top = now;
                    now->right = addedNode;
                    return;
                }
                else now = now->right;
            }
            if (now->name > a) {
                if (now->left==NULL) {
                    addedNode->name=a;
                    addedNode->top=now;
                    now->left=addedNode;
                    return;
                }
                else now=now->left;
            }
            if(now->name==a){
                now->counter++;
                delete addedNode;
                return;
            }
       }
    }
}

void inorder(node *v){
    if(v->left!=NULL){
        inorder(v->left);
    }
    for(uint i=0;i<v->counter;++i)
        cout<<v->name<<"\n";
    if(v->right!=NULL){
        inorder(v->right);
    }
}

int main()
{
    std::ios_base::sync_with_stdio(false);
    std::cout.tie(nullptr);
    std::cin.tie(nullptr);

    uint n;
    string *number=new string ;
    cin>>n;
    for(uint i=1;i<=n;i++){
        cin>>*number;
        add_node(*number);
    }
    inorder(root);
    delete root;
    delete number;
    return 0;
}