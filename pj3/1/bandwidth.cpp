#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;
using std::pow;

void test_bandwidth(int mype, string unit){
	int size = unit=="KB"? 1024 : 1024*(unit=="MB"? 1024: 1024*1024);
	size /= sizeof(int);
	int times = unit=="GB"? 1 : 10*(unit=="MB"? 1: 10);
	vector<int> sd_buf(size), rv_buf(size);
	
	auto t1 = steady_clock::now();

	for(auto i=0; i<times; ++i){
		if(mype == 0){
			MPI_Send(sd_buf.data(), size, MPI_INT, 1, 0, MPI_COMM_WORLD);
		}
		else{
			MPI_Recv(rv_buf.data(), size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	//timing
	if(mype == 0){
		auto t2 = steady_clock::now();
		duration<double> t = (t2-t1)/times;
		auto data_size = size*sizeof(int)/pow(2, 30);
		cout << "Average bandwidth is: " << data_size/t.count() <<" GB/s." << endl;
	}
}

int main(int argc, char** argv){
	string unit{argv[1]};

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
	test_bandwidth(mype, unit);

	//close mpi
	MPI_Finalize();

	return 0;
}
