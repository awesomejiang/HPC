#include <cstdlib>
#include "nbody.h"

int main(int argc, char **argv){
	double dt = 0.2;
	int n_iters = 1000;
	char fname[20] = "nbody.dat";

	System s(atoi(argv[1]));
	s.run_simulation(dt, n_iters, fname);
}