#ifndef NBODY_H
#define NBODY_H

#include <cmath>
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <cstdio>
#include <vector>
#include "vec3.h"

typedef Vec3<double> Vec;

class Body{
public:
	Body();
	Body(Vec r, Vec v, double m);

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

	void init_system();
	void compute_forces(double dt);

};

double get_time();

#endif