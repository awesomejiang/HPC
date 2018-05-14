#include <iostream>
#include <mpi.h>
#include <cmath>
#include <numeric>
#include <vector>
#include "parallel.h"

Parallel::Parallel(long n, int nprocs, int mype)
: Poisson(n), nprocs{nprocs}, mype{mype}, x_offset{mype*n/nprocs},
  x_range{(mype+1)*n/nprocs - x_offset} {}


vec Parallel::matvec(vec const &w){
	//compute values of ghost cells
	vec ghost_l(n, 0.0), ghost_r(n, 0.0);

	if(mype != 0)
		MPI_Recv(ghost_l.data(), n, MPI_DOUBLE, mype-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	if(mype != nprocs-1)
		MPI_Send(w.data()+x_range*n-n, n, MPI_DOUBLE, mype+1, 0, MPI_COMM_WORLD);

	if(mype != nprocs-1)
		MPI_Recv(ghost_r.data(), n, MPI_DOUBLE, mype+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	if(mype != 0)
		MPI_Send(w.data(), n, MPI_DOUBLE, mype-1, 1, MPI_COMM_WORLD);

	// loop to compute matvec
	vec v(x_range*n);
	for(auto i=0; i<x_range*n; ++i){
		// Physical 2D x-coordinate
		long x = i%n;
		// compute related diagonals
		vec diags = {
			// Far left diagonal
			-1.0 * ((i-n<0)? ghost_l[x]: w[i-n]),
			// left diagonal
			x? -1.0 * w[i-1]: 0.0,
			// Main diagonal
			4.0 * w[i],
			// Right diagonal
			(x != n-1)? -1.0 * w[i+1]: 0.0,
			// Far right diagonal
			-1.0 * ((i+n<x_range*n)? w[i+n]: ghost_r[x])
		};
		v[i] = std::accumulate(diags.begin(), diags.end(), 0.0);
	}

	return v;
}

double Parallel::dotp(vec const &a, vec const &b){
	double sum_local = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
	double sum{0.0};

	//broadcast reduced sum to all other ranks
	MPI_Allreduce(&sum_local, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	return sum;
}

vec Parallel::axpy(double a, vec const &w, double b, vec const &v){
	vec res(x_range*n);
	for(auto i=0; i<x_range*n; ++i)
		res[i] = a * w[i] + b * v[i];

	return res;
}

vec Parallel::fill_b(){
	vec res(x_range*n);
	for(auto i=0; i<x_range; ++i)
		for(auto j=0; j<n; ++j)
			res[i*n+j] = find_b(x_offset+i, j);

	return res;
}

vec Parallel::solve(){
	//init
	auto b = fill_b();
	vec x(x_range*n, 0.0);

	// r = -A*x + b
	auto r = axpy(-1.0, matvec(x), 1.0, b);
	// p = r;
	auto p = r;
	// rsold = r' * r;
	auto rsold = dotp(r, r);
	// z
	vec z;

	long iter;
	for(iter = 0; iter < n*n; ++iter){
		//z = A * p;
		z = matvec(p);
		//alpha = rsold / (p' * z);
		auto alpha = rsold/dotp(p, z);
		//x = x + alpha * p;
		x = axpy(1.0, x, alpha, p);
		//r = r - alpha * z;
		r = axpy(1.0, r, -alpha, z);

		double rsnew = dotp(r,r);
		int c_local = std::sqrt(rsnew) < 1.0e-10;
		int c;

		//broadcast converged result to others
		MPI_Allreduce(&c_local, &c, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		//converged is 1 iff every ranks is converged.
		if(c == nprocs)
			break;

		//p = (rsnew / rsold) * p + r;
		p = axpy(rsnew/rsold, p, 1.0, r);

		rsold = rsnew;
	}
		
	if(mype == 0)
		std::cout << "CG converged in " << iter << " iterations." << std::endl;
	return x;
}
