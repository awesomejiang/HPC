#include <cstdlib>
#include <iostream>
#include <omp.h>

#include "include/advection.h"

using namespace advection;
using namespace std;

/*
void print_args(char **argv){
	int N = atoi(argv[1]), NT = atoi(argv[2]),
		thread = atoi(argv[7]);
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
		 << "Thread used:\t" << thread << "\n"
		 << "Estimated memory usage:\t" << N*N*2*sizeof(double)/1024
		 << "kb" <<endl;
}
*/
/*
	test args: 
	400 20000 1.0 1.0e6 5.0e-7 2.95e-7 4
*/
int main(int argc, char **argv){
	Driver d(argc, argv);
	d.run();

	return 0;
}