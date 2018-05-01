#include <mpi.h>
#include <iostream>
#include <stdexcept>
#include <chrono>

using namespace std;
using namespace std::chrono;


void test_latency(int mype, int times){
	int sd_buf, rv_buf;
	auto t1 = steady_clock::now();

	for(auto i=0; i<times; ++i){
		if(mype == 0){
			int buf = 0;
			MPI_Send(&sd_buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
			MPI_Recv(&rv_buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		else{
			int buf = 0;
			MPI_Recv(&rv_buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(&sd_buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		}
		//MPI_Barrier(MPI_COMM_WORLD);
	}

	//timing
	if(mype == 0){
		auto t2 = steady_clock::now();
		double latency = static_cast<double>(duration_cast<microseconds>(t2-t1).count())/times;
		cout << "Average roundtime latency is: " << latency <<" us." << endl;
	}
}

int main(int argc, char** argv){
	int times = atoi(argv[1]);

	//init mpi
	int nprocs, mype, stat;
	MPI_Init(&argc, &argv);
	stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	if(stat != MPI_SUCCESS)
		throw runtime_error("Cannot set nprocs.");
	if(nprocs != 2)
		throw runtime_error("This test should only run under 2 rank case.");
	stat = MPI_Comm_rank(MPI_COMM_WORLD, &mype);
	if(stat != MPI_SUCCESS)
		throw runtime_error("Cannot set mype.");

	//test!
	test_latency(mype, times);

	//close mpi
	MPI_Finalize();

	return 0;
}