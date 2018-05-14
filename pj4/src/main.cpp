#include <cstdlib>
#include <iostream>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <mpi.h>
#include <string>
#include "serial.h"
#include "parallel.h"

using namespace std;

void run_serial(long n){
	cout << "Running Serial CG solve..." << endl;

	auto t1 = chrono::steady_clock::now();

	// solve
	Serial s(n);
	auto v = s.solve();

	auto t2 = chrono::steady_clock::now();		
	std::cout << "Serial Running time: "
		 << chrono::duration_cast<chrono::milliseconds>(t2-t1).count()
		 << " ms" << endl;

	//write file
	ofstream of("serial.out", ios::trunc | ios::in | ios::out | ios::binary);
	of.write(reinterpret_cast<char *>(v.data()), v.size()*sizeof(double));
	of.close();

}

void run_parallel(long n){
	auto t1 = chrono::steady_clock::now();

	int nprocs = 1; int mype = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &mype);

	if(mype == 0)
		cout << "Running Parallel CG solve..." << endl;

	// solve
	Parallel p(n, nprocs, mype);
	auto v = p.solve();

	if(mype == 0){
		auto t2 = chrono::steady_clock::now();		
		std::cout << "Parallel Running time: "
			 << chrono::duration_cast<chrono::milliseconds>(t2-t1).count()
			 << " ms" << endl;
	}

	//write file
	MPI_File fh;
	MPI_Offset offset = mype*n/nprocs*n * sizeof(double);
	MPI_File_open(MPI_COMM_WORLD, "parallel.out",
		MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	MPI_File_seek(fh, offset, MPI_SEEK_SET);
	MPI_File_write(fh, v.data(), v.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);

}

int main(int argc, char** argv){
	if(string{argv[2]} == "serial")
		run_serial(atoi(argv[1]));
	else if(string{argv[2]} == "parallel"){
		ofstream of("parallel.out");
		of.close();
		MPI_Init(&argc, &argv);
		run_parallel(atoi(argv[1]));
		MPI_Finalize();
	}
	else
		throw runtime_error("Unrecognized input argument.");

	return 0;
}