#include <mpi.h>
#include <algorithm>

#include "advection.h"

using namespace advection;


Advection_mpi_non_blocking::
Advection_mpi_non_blocking(int N, int NT, double L, double T, double u, double v, int mype, int k)
	: Advection_mpi_blocking(N, NT, L, T, u, v, mype, k) {}

void Advection_mpi_non_blocking::
sync(){
	MPI_Request r;
	line<double> u_ghost = curr_mx[0], d_ghost = curr_mx[N-1], 
		l_ghost, r_ghost;
	std::transform(curr_mx.begin(), curr_mx.end(), std::back_inserter(l_ghost),
		[](auto l){return l.front();});
	std::transform(curr_mx.begin(), curr_mx.end(), std::back_inserter(r_ghost),
		[](auto l){return l.back();});
	//cal destinations
	int up = (mype-k+k*k)%(k*k), down = (mype+k)%(k*k),
		left = mype/k*k+(mype+3)%k, right = mype/k*k+(mype+1)%k;
	//send msg
	MPI_Isend(u_ghost.data(), N, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &r);
	MPI_Isend(r_ghost.data(), N, MPI_DOUBLE, right, 1, MPI_COMM_WORLD, &r);
	MPI_Isend(d_ghost.data(), N, MPI_DOUBLE, down, 2, MPI_COMM_WORLD, &r);
	MPI_Isend(u_ghost.data(), N, MPI_DOUBLE, left, 3, MPI_COMM_WORLD, &r);
	//recv msg
	MPI_Irecv(ghost_cells[2].data(), N, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &r);
	MPI_Irecv(ghost_cells[3].data(), N, MPI_DOUBLE, left, 1, MPI_COMM_WORLD, &r);
	MPI_Irecv(ghost_cells[0].data(), N, MPI_DOUBLE, up, 2, MPI_COMM_WORLD, &r);
	MPI_Irecv(ghost_cells[1].data(), N, MPI_DOUBLE, right, 3, MPI_COMM_WORLD, &r);

	MPI_Wait(&r, MPI_STATUS_IGNORE);
}
