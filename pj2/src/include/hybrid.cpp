#include <mpi.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iterator>

#include "advection.h"

using namespace advection;

Advection_hybrid::
Advection_hybrid(int N, int NT, double L, double T, double u, double v, int mype, int k)
	: Advection_mpi_blocking(N, NT, L, T, u, v, mype, k) {}

void Advection_hybrid::
init_gaussian(double sig_x, double sig_y, double x0, double y0){
	#pragma omp parallel for default(none) shared(sig_x, sig_y, x0, y0)
	for(auto i=0; i<N; ++i)
		for(auto j=0; j<N; ++j)
			curr_mx[i][j] = 
				std::exp(-0.5*(pow((i*delta_x-x0)/sig_x, 2) + pow((j*delta_x-y0)/sig_y, 2)));
}

void Advection_hybrid::run(){
	for(auto t=0; t<NT; ++t){
		sync();
		#pragma omp parallel for default(none)
		for(auto i=0; i<N; ++i)
			for(auto j=0; j<N; ++j)
				next_mx[i][j] = update_value(i, j);
		std::swap(curr_mx, next_mx);
	}
}

void Advection_hybrid::run(std::string file){
	for(auto t=0; t<NT; ++t){
		sync();
		#pragma omp parallel for default(none)
		for(auto i=0; i<N; ++i)
			for(auto j=0; j<N; ++j)
				next_mx[i][j] = update_value(i, j);
		
		//write
		if(t%(NT/20) == 0){
			MPI_File fh;
			MPI_Offset offset;
			MPI_File_open(MPI_COMM_WORLD, file.c_str(), 
				MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
			for(auto l=0; l<N; ++l){
				offset = sizeof(double)*(t/(NT/20)*N*k*N*k + (mype/k*N+l)*N*k + mype%k*N);
				MPI_File_seek(fh, offset, MPI_SEEK_SET);
				MPI_File_write(fh, curr_mx[l].data(), N, MPI_DOUBLE, MPI_STATUS_IGNORE);
			}
			MPI_File_close(&fh);
		}
		std::swap(curr_mx, next_mx);
	}
}