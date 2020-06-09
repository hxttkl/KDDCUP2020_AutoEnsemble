#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <ctime>
#include <iostream>
#include <fstream>
#include <set>
#include <map>
#include <unordered_map>
#include <algorithm>
using namespace std;


typedef long long int64;
typedef pair<int,int> PII;
struct PAIR {
	int a, b;
	PAIR(int a0, int b0) { a=a0; b=b0; }
};
bool operator<(const PAIR &x, const PAIR &y) {
	if (x.a==y.a) return x.b<y.b;
	else return x.a<y.a;
}
bool operator==(const PAIR &x, const PAIR &y) {
	return x.a==y.a && x.b==y.b;
}
map<PAIR, int> edge;
int n,m;
int *str;
int *dst;
float *weight;
int GS;
fstream fin, fout; // input and output files
int init(int argc, char *argv[]) {
	// open input, output files
	sscanf(argv[1],"%d",&GS);
	printf("%d",GS);
	fin.open(argv[2], fstream::in);
	fout.open(argv[3], fstream::out | fstream::binary);
	if (fin.fail()) {
		cerr << "Failed to open file " << argv[2] << endl;
		return 0;
	}
	if (fout.fail()) {
		cerr << "Failed to open file " << argv[3] << endl;
		return 0;
	}
    int k = 0;
	fin>>n>>m>>k;
    printf("%d %d\n",n,m);
    str = (int*)calloc(m,sizeof(int));
    dst = (int*)calloc(m,sizeof(int));
    weight = (float*)calloc(m,sizeof(float));
	for (int i=0;i<m;i++) {
		int a,b;
        float c;
		fin>>a>>b>>c;
        str[i]=b;
        dst[i]=a;
        weight[i]=c;
        edge[PAIR(a,b)]=1;
    }
	return 1;
}
void writeResults() {
    printf("%d",edge[PAIR(0,4186)]);
	for (int i=0;i<m;i++) {
        if(edge[PAIR(str[i],dst[i])]!=1){
            fout << str[i];
            fout << " ";
            fout << dst[i];
            fout << " ";
            fout << weight[i];
            fout << endl;
        }
	}
	fout.close();
}
int main(int argc, char *argv[]) {
	if (!init(argc, argv)) {
		cerr << "Stopping!" << endl;
		return 0;
	}
	writeResults();
	return 0;
}


