#ifndef POISSON_H
#define POISSON_H

#include <vector>

using vec = std::vector<double>;

class Poisson{
public:
	// n = physical dimension
	Poisson(long n);

	// A*w
	virtual vec matvec(vec const &w) = 0;
	// a*b
	virtual double dotp(vec const &a, vec const &b) = 0;
	// a*w + b*v
	virtual vec axpy(double a, vec const &w, double b, vec const &v) = 0;
	// main solver function
	virtual vec solve() = 0;
	// init b[]
	virtual vec fill_b() = 0;

	// b[i*n+j]
	double find_b(long i, long j);

	long n;
};

#endif