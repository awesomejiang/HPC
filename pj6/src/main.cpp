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

// the usage 
void usage(char* program){
    fprintf(stderr, "\nProgram Usage:\n%s\n", program);
    fprintf(stderr, "     [ -n     128       ]    number of particles, default: 128\n");
    fprintf(stderr, "     [ -d     0.2       ]    time interval, default: 0.2\n");
    fprintf(stderr, "     [ -i     1000      ]    timesteps for iteration, default: 1000\n");
    fprintf(stderr, "     [ -o   nbody.dat   ]    moving steps, default: dat/nbody.dat\n");
    fprintf(stderr, "     [ -t      8        ]    threads for openmp, default: 8\n");
    fprintf(stderr, "     [ -h               ]    display this information\n");
    fprintf(stderr, "\n");
    exit(1);
}

// function to get system arguments from command line
void get_args(int argc, char** argv, int* n, double* dt, int *iter, char** fname, int* thread){	
    char ch;
    while (( ch = getopt(argc, argv, "n:d:i:o:t:h")) != -1){
        switch (ch){
        case 'n':
            *n = atoi(optarg);break;
        case 'd':
            *dt = atof(optarg);break;
        case 'i':
            *iter = atoi(optarg);break;
        case 'o':
            *fname = optarg;break;
        case 't':
            *thread = atoi(optarg);break;
        case 'h':
            usage(argv[0]);
        case '?':
            usage(argv[0]);
        }
    }
}

int main(int argc, char **argv){
	int n = 128;
	double dt = 0.2;
	int n_iters = 1000;
	char *fname = (char*)"nbody.dat";
	int thread = 8;
	get_args(argc, argv, &n, &dt, &n_iters, &fname, &thread);
    fname = (char*)(std::string("dat/") + std::string(fname)).c_str();

	#ifdef OPENMP_ON

	omp_set_num_threads(thread);

	#endif

	#ifdef MPI_ON

    // Have to truncate output file manully
    std::ofstream of(fname);
    of.close();
	MPI_Init(&argc, &argv);

	#endif

	System s(n);
	s.run_simulation(dt, n_iters, fname);

	#ifdef MPI_ON

	MPI_Finalize();	

	#endif
}