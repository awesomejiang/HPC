#include <cstdlib>
#include <cassert>
#include <unistd.h>
#include <fstream>
#include "nbody.h"
#include <string>

#ifdef MPI_ON

#include <mpi.h>

#endif

#ifdef OPENMP_ON

#include <omp.h>

#endif

int main(int argc, char **argv){

	assert(argc == 5);
	int n = atoi(argv[1]);
	int n_iters = atoi(argv[2]);
	int thread = atoi(argv[3]);
	char *fname = argv[4];	
/*
	int n = 102400;
	double dt = 0.2;
	int n_iters = 10;m
	int thread = 1;
	char *fname = (char*)"dat/nbody.dat";
*/

	#ifdef OPENMP_ON

	omp_set_num_threads(thread);

	#endif

	#ifdef MPI_ON

	// Have to truncate output file manully
	MPI_Init(&argc, &argv);
	int mype;
	MPI_Comm_rank(MPI_COMM_WORLD, &mype);
	if(mype == 0){
		std::ofstream of(fname);
		of.close();
	}

	#endif

	System s(n);
	s.run_simulation(n_iters, fname);

	#ifdef MPI_ON



	MPI_Finalize();	

	#endif
}
