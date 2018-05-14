#include <iostream>
#include <cmath>
#include <numeric>
#include "serial.h"

Serial::Serial(long n): Poisson(n) {}

vec Serial::matvec(vec const &w){
	vec v(n*n);

	for(auto i=0; i<n*n; ++i){
		// Physical 2D x-coordinate
		long x = i%n;

		vec diags = {
			// Far left diagonal
			(i >= n)? -1.0 * w[i-n]: 0.0,
			// left diagonal
			x? -1.0 * w[i-1]: 0.0,
			// Main diagonal
			4.0 * w[i],
			// Right diagonal
			(x != n-1)? -1.0 * w[i+1]: 0.0,
			// Far right diagonal
			(i < n*n - n)? -1.0 * w[i+n]: 0.0
		};
		v[i] = std::accumulate(diags.begin(), diags.end(), 0.0);
	}

	return v;
}

double Serial::dotp(vec const &a, vec const &b){
	return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

vec Serial::axpy(double a, vec const &w, double b, vec const &v){
	vec res(n*n);
	for(auto i=0; i<n*n; ++i)
		res[i] = a * w[i] + b * v[i];

	return res;
}

vec Serial::fill_b(){
	vec res(n*n);
	for(auto i=0; i<n; ++i)
		for(auto j=0; j<n; ++j)
			res[i*n+j] = find_b(i, j);

	return res;
}

vec Serial::solve(){
	//init
	vec x(n*n);
	auto b = fill_b();

	// r = -A*x + b
	auto r = axpy(-1.0, matvec(x), 1.0, b);
	// p = r;
	auto p = r;
	// rsold = r' * r;
	auto rsold = dotp(r, r);
	// z
	vec z;

	for(long iter = 0; iter < n*n; ++iter){
		//z = A * p;
		z = matvec(p);
		//alpha = rsold / (p' * z);
		auto alpha = rsold/dotp(p, z);
		//x = x + alpha * p;
		x = axpy(1.0, x, alpha, p);
		//r = r - alpha * z;
		r = axpy(1.0, r, -alpha, z);

		double rsnew = dotp(r,r);
		if(std::sqrt(rsnew) < 1.0e-10)
			break;

		//p = (rsnew / rsold) * p + r;
		p = axpy(rsnew/rsold, p, 1.0, r);

		rsold = rsnew;
	}
		
	std::cout << "CG converged in " << n*n << " iterations." << std::endl;
	return x;
}
