#ifndef PARALLEL_H
#define PARALLEL_H

#include "poisson.h"

class Parallel: public Poisson{
public:
	Parallel(long n, int nprocs, int mype);

	vec solve() override;

private:
	vec matvec(vec const &w) override;
	double dotp(vec const &a, vec const &b) override;
	vec axpy(double a, vec const &w, double b, vec const &v) override;
	vec fill_b() override;

	int nprocs, mype;
	long x_offset, x_range;
};

#endif