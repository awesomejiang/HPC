#ifndef NBODY_H
#define NBODY_H

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <vector>
#include <string.h>
#include "vec3.h"

#ifdef MPI_ON

#include <mpi.h>

#endif

#ifdef OPENMP_ON

#include <omp.h>

#endif

typedef Vec3<double> Vec;

class Body{
public:
	Body(): r(0, 0, 0), v(0, 0, 0), m(0) {}
	Body(Vec r, Vec v, double m): r(r), v(v), m(m) {}

	Vec r, v;
	double m;
};

class System{
public:
	System();
	System(int n_bodies);

	void run_simulation(double dt, int n_iters, char *fname);

private:
	int n_bodies;
	std::vector<Body> bodies;

	#ifdef MPI_ON

	int nprocs, mype;
	std::vector<Body> in_buffer, out_buffer;
	MPI_Datatype MPI_BODY;

	#endif

	void init_system();
	void init_bodies();
	void compute_forces(double dt);

	/* init conditions */
	void uniform_random();
	void three_body();
	void shpere();
	void galaxy();

};

double get_time();

#endif
