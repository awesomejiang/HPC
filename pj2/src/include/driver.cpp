#include <cstdlib>
#include <string>
#include <stdexcept>
#include <cmath>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <mpi.h>
#include "advection.h"

using namespace advection;

using std::runtime_error;
using std::cout;
using std::endl;


Driver::Driver(int argc, char**argv)
	: argc(argc), argv(argv), 
	N(atoi(argv[1])), NT(atoi(argv[2])),
	thread(atoi(argv[7])),L(atof(argv[3])), 
	T(atof(argv[4])), u(atof(argv[5])), v(atof(argv[6])),
	mode(std::string{argv[8]}) {
		//check Courant stability condition
		double delta_x = L/N, delta_t = T/NT;
		if(delta_t > delta_x/(std::sqrt(2)*std::hypot(u, v)))
			throw runtime_error("Args are instable.");
}


void Driver::run(){
	if(mode == "serial") serial();
	else if(mode == "threads") threads();
	else if(mode == "mpi_blocking") mpi_blocking();
	else if(mode == "mpi_non_blocking") mpi_non_blocking();
	else if(mode == "hybrid") hybrid();
	else throw runtime_error("");
}


void Driver::serial(){
	//check thread is 1
	if(thread != 1)
		throw runtime_error("Thread has to be 1 in serial version.");

	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	//run simulation
	Advection_serial ad{N ,NT, L, T, u, v};
	ad.init_gaussian(L/4, L/4, L/2, L/2);
	if(argc == 10 && std::string{argv[9]} == "silence")
		ad.run();
	else
		ad.run("serial.out");

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	cout << "Running time: "
		 << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()
		 << " ms" << endl;
}


void Driver::threads(){
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	//set omp threads
	omp_set_num_threads(thread);

	//run simulation
	Advection_threads ad{N ,NT, L, T, u, v};
	ad.init_gaussian(L/4, L/4, L/2, L/2);
	if(argc == 10 && std::string{argv[9]} == "silence")
		ad.run();
	else
		ad.run("threads.out");

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	cout << "Running time: "
		 << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()
		 << " ms" << endl;
}


void Driver::mpi_blocking(){
	//check if thread == 1
	if(thread != 1)
		throw runtime_error("Thread has to be 1 in mpi version.");

	//create/clear output file before mpi init
	std::ofstream f("mpi_blocking.out", std::ofstream::trunc);
	f.close();

	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	
	//mpi init
	int nprocs, mype, stat;
	MPI_Init(&argc, &argv);
	stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* return number of procs */
	if(stat != MPI_SUCCESS)
		throw runtime_error("Cannot set nprocs.");

	int k = std::sqrt(nprocs);
	if(k*k != nprocs) // check if rank is sqare number
		throw runtime_error("Mpi rank has to be sqare number.");
	else if(N%k != 0)
		throw runtime_error("N cannot be divided by sqrt of rank.");

	stat = MPI_Comm_rank(MPI_COMM_WORLD, &mype); /* my integer proc id */
	if(stat != MPI_SUCCESS)
		throw runtime_error("Cannot set mype.");

	//run simulation
	Advection_mpi_blocking ad{N/k ,NT, L/k, T, u, v, mype, k};
	ad.init_gaussian(L/4, L/4, L/2-mype/k*L/k, L/2-(mype%k)*L/k);
	if(argc == 10 && std::string{argv[9]} == "silence")
		ad.run();
	else
		ad.run("mpi_blocking.out");

	if(mype == 0){
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		cout << "Running time: "
			 << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()
			 << " ms" << endl;
	}

	//mpi finalize
	MPI_Finalize();
}


void Driver::mpi_non_blocking(){
	//check if thread == 1
	if(thread != 1)
		throw runtime_error("Thread has to be 1 in mpi version.");

	//create/clear output file before mpi init
	std::ofstream f("mpi_non_blocking.out", std::ofstream::trunc);
	f.close();

	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	
	//mpi init
	int nprocs, mype, stat;
	MPI_Init(&argc, &argv);
	stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* return number of procs */
	if(stat != MPI_SUCCESS)
		throw runtime_error("Cannot set nprocs.");

	int k = std::sqrt(nprocs);
	if(k*k != nprocs) // check if rank is sqare number
		throw runtime_error("Mpi rank has to be sqare number.");
	else if(N%k != 0)
		throw runtime_error("N cannot be divided by sqrt of rank.");

	stat = MPI_Comm_rank(MPI_COMM_WORLD, &mype); /* my integer proc id */
	if(stat != MPI_SUCCESS)
		throw runtime_error("Cannot set mype.");

	//run simulation
	Advection_mpi_non_blocking ad{N/k ,NT, L/k, T, u, v, mype, k};
	ad.init_gaussian(L/4, L/4, L/2-mype/k*L/k, L/2-(mype%k)*L/k);
	if(argc == 10 && std::string{argv[9]} == "silence")
		ad.run();
	else
		ad.run("mpi_non_blocking.out");

	if(mype == 0){
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		cout << "Running time: "
			 << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()
			 << " ms" << endl;
	}

	//mpi finalize
	MPI_Finalize();
}


void Driver::hybrid(){
	//set omp threads
	omp_set_num_threads(thread);

	//create/clear output file before mpi init
	std::ofstream f("hybrid.out", std::ofstream::trunc);
	f.close();

	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	
	//mpi init
	int nprocs, mype, stat;
	MPI_Init(&argc, &argv);
	stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* return number of procs */
	if(stat != MPI_SUCCESS)
		throw runtime_error("Cannot set nprocs.");

	int k = std::sqrt(nprocs);
	if(k*k != nprocs) // check if rank is sqare number
		throw runtime_error("Mpi rank has to be sqare number.");
	else if(N%k != 0)
		throw runtime_error("N cannot be divided by sqrt of rank.");

	stat = MPI_Comm_rank(MPI_COMM_WORLD, &mype); /* my integer proc id */
	if(stat != MPI_SUCCESS)
		throw runtime_error("Cannot set mype.");

	//run simulation
	Advection_hybrid ad{N/k ,NT, L/k, T, u, v, mype, k};
	ad.init_gaussian(L/4, L/4, L/2-mype/k*L/k, L/2-(mype%k)*L/k);
	if(argc == 10 && std::string{argv[9]} == "silence")
		ad.run();
	else
		ad.run("hybrid.out");

	if(mype == 0){
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		cout << "Running time: "
			 << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()
			 << " ms" << endl;
	}

	//mpi finalize
	MPI_Finalize();
}