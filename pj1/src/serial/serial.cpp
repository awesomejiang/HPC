#include <cstdlib>
#include <ctime>
#include <iostream>

#include "advection.h"

using namespace advection;
using namespace std;

void print_args(char **argv){
	int N = atoi(argv[1]), NT = atoi(argv[2]);
	double L = atof(argv[3]), T = atof(argv[4]),
		 u = atof(argv[5]), v = atof(argv[6]);
	cout << "Maxtrix Dimension(N):\t" << N << "\n"
		 << "Number of timesteps(NT):\t" << NT << "\n"
		 << "Physical Domain Length(L):\t" << L << "\n"
		 << "Total Physical TImespan(T):\t" << T << "\n"
		 << "X velocity Scalar(u):\t" << u << "\n"
		 << "Y velocity Scalar(v):\t" << v << "\n"
		 << "X Gaussian spread(sigma_x)\t" << L/4 << "\n"
		 << "Y Gaussian spread(sigma_y)\t" << L/4 << "\n"
		 << "Estimated memory usage:\t" << N*N*2*sizeof(double)/1024
		 << "kb" <<endl;
}

/*
	test args: 
	400 20000 1.0 1.0e6 5.0e-7 2.95e-7
*/
int main(int argc, char **argv){
	time_t t1 = time(nullptr);
	int N = atoi(argv[1]), NT = atoi(argv[2]);
	double L = atof(argv[3]), T = atof(argv[4]),
		 u = atof(argv[5]), v = atof(argv[6]);

	print_args(argv);

	Advection ad{N ,NT, L, T, u, v};
	ad.init_gaussian(L/4, L/4);
	ad.run("serial.out");

	time_t t2 = time(nullptr);
	cout<<"Serial running time:\t"<<t2-t1<<" s."<<endl;

	return 0;
}
