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

Advection_mpi_blocking::
Advection_mpi_blocking(int N, int NT, double L, double T, double u, double v, int mype, int k)
	: Advection_common(N, NT, L, T, u, v), mype(mype), k(k), 
	  ghost_cells(matrix<double>(4, line<double>(N, 0))) {}

void Advection_mpi_blocking::
init_gaussian(double sig_x, double sig_y, double x0, double y0){
	for(auto i=0; i<N; ++i)
		for(auto j=0; j<N; ++j)
			curr_mx[i][j] = 
				std::exp(-0.5*(pow((i*delta_x-x0)/sig_x, 2) + pow((j*delta_x-y0)/sig_y, 2)));
}

void Advection_mpi_blocking::
Advection_mpi_blocking::run(){
	for(auto t=0; t<NT; ++t){
		sync();
		for(auto i=0; i<N; ++i)
			for(auto j=0; j<N; ++j)
				next_mx[i][j] = update_value(i, j);
		std::swap(curr_mx, next_mx);
	}
}

void Advection_mpi_blocking::
Advection_mpi_blocking::run(std::string file){
	for(auto t=0; t<NT; ++t){
		sync();
		for(auto i=0; i<N; ++i)
			for(auto j=0; j<N; ++j)
				next_mx[i][j] = update_value(i, j);

		//write
		if(t%(NT/20) == 0){
			MPI_File fh;
			MPI_Offset offset;
			MPI_File_open(MPI_COMM_WORLD, const_cast<char*>(file.c_str()),
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

double Advection_mpi_blocking::
update_value(int i, int j){
	//periodic conditions
	double up = curr_mx[(i-1+N)%N][j],
		down = curr_mx[(i+1)%N][j],
		left = curr_mx[i][(j-1+N)%N],
		right = curr_mx[i][(j+1)%N];
	if(i == 0) up = ghost_cells[0][j];
	if(i == N-1) down = ghost_cells[2][j];
	if(j == 0) left = ghost_cells[3][i];
	if(j == N-1) right = ghost_cells[1][i];

	//"eq 6"
	return 0.25*(up+down+left+right)
		- delta_t/(2*delta_x)*(u*(down-up)+v*(right-left));		
}

void Advection_mpi_blocking::sync_up(){
	line<double> buf = curr_mx[0];
	int up = (mype-k+k*k)%(k*k), down = (mype+k)%(k*k);
	if(mype/k == 0){
		MPI_Send(buf.data(), N, MPI_DOUBLE, up, 0, MPI_COMM_WORLD);
		MPI_Recv(ghost_cells[2].data(), N, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	else{
		MPI_Recv(ghost_cells[2].data(), N, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(buf.data(), N, MPI_DOUBLE, up, 0, MPI_COMM_WORLD);
	}
}

void Advection_mpi_blocking::sync_down(){
	line<double> buf = curr_mx[N-1];
	int up = (mype-k+k*k)%(k*k), down = (mype+k)%(k*k);
	if(mype/k == 0){
		MPI_Send(buf.data(), N, MPI_DOUBLE, down, 0, MPI_COMM_WORLD);
		MPI_Recv(ghost_cells[0].data(), N, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	else{
		MPI_Recv(ghost_cells[0].data(), N, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(buf.data(), N, MPI_DOUBLE, down, 0, MPI_COMM_WORLD);
	}
}

void Advection_mpi_blocking::sync_left(){
	line<double> buf;
	std::transform(curr_mx.begin(), curr_mx.end(), std::back_inserter(buf),
		[](auto l){return l.front();});
	int left = mype/k*k+(mype+3)%k, right = mype/k*k+(mype+1)%k;
	if(mype%k == 0){
		MPI_Send(buf.data(), N, MPI_DOUBLE, left, 0, MPI_COMM_WORLD);
		MPI_Recv(ghost_cells[1].data(), N, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	else{
		MPI_Recv(ghost_cells[1].data(), N, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(buf.data(), N, MPI_DOUBLE, left, 0, MPI_COMM_WORLD);
	}
}

void Advection_mpi_blocking::sync_right(){
	line<double> buf;
	std::transform(curr_mx.begin(), curr_mx.end(), std::back_inserter(buf),
		[](auto l){return l.back();});
	int left = mype/k*k+(mype+3)%k, right = mype/k*k+(mype+1)%k;
	if(mype%k == 0){
		MPI_Send(buf.data(), N, MPI_DOUBLE, right, 0, MPI_COMM_WORLD);
		MPI_Recv(ghost_cells[3].data(), N, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	else{
		MPI_Recv(ghost_cells[3].data(), N, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(buf.data(), N, MPI_DOUBLE, right, 0, MPI_COMM_WORLD);
	}
}

void Advection_mpi_blocking::
sync(){
	sync_up();
	sync_down();
	sync_left();
	sync_right();
}
