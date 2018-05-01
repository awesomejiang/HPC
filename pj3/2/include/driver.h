#include <mpi.h>
#include <string>
#include <cstdlib>
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <iostream>
#include "julia.h"

using std::runtime_error;

class Driver{
public:
	Driver(int argc, char** argv): argc{argc}, argv{argv},
		xmin{-1.5}, xmax{1.5}, ymin{-1.0}, ymax{1.0},
		Nx{12000}, Ny{12000}, mode{argv[1]}{
			if(argc == 3) //read dynamic chunks arg
				chunks = atoi(argv[2]);
	}

	void run(){
		if(mode == "serial")
			driver_serial();
		else if(mode == "static")
			driver_static();
		else if(mode == "dynamic")
			driver_dynamic();
		else
			throw runtime_error("Unrecognized mode.");
	}

private:
	int argc, Nx, Ny, chunks;
	char **argv;
	double xmin, xmax, ymin , ymax;
	std::string mode;
	int nprocs, mype, stat;

	void driver_serial(){
		auto t1 = std::chrono::steady_clock::now();

		//create/clear outputfile
		std::ofstream f("serial.out", std::ofstream::trunc);
		f.close();

		Julia j(xmin, xmax, ymin, ymax, Nx, Ny);
		write_serial("serial.out", j.run(), 0);

		auto t2 = std::chrono::steady_clock::now();
		std::cout << "Running time: "
			 << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
			 << " ms" << std::endl;
	}

	void driver_static(){
		auto t1 = std::chrono::steady_clock::now();

		//create/clear outputfile
		std::ofstream f("static.out", std::ofstream::trunc);
		f.close();
		//init mpi
		MPI_Init(&argc, &argv);
		stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
		if(stat != MPI_SUCCESS)
			throw runtime_error("Cannot set nprocs.");
		stat = MPI_Comm_rank(MPI_COMM_WORLD, &mype);
		if(stat != MPI_SUCCESS)
			throw runtime_error("Cannot set mype.");

		//test!
		auto dx = (xmax-xmin)/nprocs;
		xmin = xmin + dx*mype;
		xmax = (mype != nprocs -1)? (xmin + dx): xmax; //care about last one
		Nx = (mype != nprocs -1)? Nx/nprocs: Nx/nprocs+Nx%nprocs;
		Julia j(xmin, xmax, ymin, ymax, Nx, Ny);
		write_mpi("static.out", j.run(), Nx*Ny*mype*sizeof(int));

		if(mype == 0){
			auto t2 = std::chrono::steady_clock::now();
			std::cout << "Running time: "
				 << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
				 << " ms" << std::endl;
		}

		//close mpi
		MPI_Finalize();
	}

	void driver_dynamic(){
		auto t1 = std::chrono::steady_clock::now();

		//create/clear outputfile
		std::ofstream f("dynamic.out", std::ofstream::trunc);
		f.close();
		//init mpi
		MPI_Init(&argc, &argv);
		stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
		if(stat != MPI_SUCCESS)
			throw runtime_error("Cannot set nprocs.");
		stat = MPI_Comm_rank(MPI_COMM_WORLD, &mype);
		if(stat != MPI_SUCCESS)
			throw runtime_error("Cannot set mype.");

		//test!
		if(mype == 0){	//boss rank
			dynamic_boss();
			auto t2 = std::chrono::steady_clock::now();
			std::cout << "Chunks/running time: " << chunks << "\t"
				 << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
				 << " ms" << std::endl;
		}
		else
			dynamic_worker();

		//close mpi
		MPI_Finalize();
	}

	void dynamic_boss(){
		vector<int> recv_buf(Nx*Ny/chunks*2);
		MPI_Status status;
		int work_cnt = 0;
		//init
		for(auto i=1; i<std::min(chunks, nprocs); ++i){
			MPI_Send(&work_cnt, 1, MPI_INT, i, i, MPI_COMM_WORLD);
			++work_cnt;
		}
		//follow up works
		for(auto i=0; i<chunks; ++i){
			//recv res
			auto size = Nx/chunks*Ny;
			recv_buf.resize(size+Nx%chunks*Ny);
			MPI_Recv(recv_buf.data(), size+Nx%chunks*Ny, MPI_INT,
				MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			if(status.MPI_TAG != chunks)
				recv_buf.resize(size);

			//write to file
			write_serial("dynamic.out", recv_buf, size*(status.MPI_TAG-1)*sizeof(int));

			//send more works
			if(work_cnt < chunks){
				MPI_Send(&work_cnt, 1, MPI_INT, status.MPI_SOURCE, work_cnt+1, MPI_COMM_WORLD);
				++work_cnt;
			}
			else	//inform worker to end
				MPI_Send(MPI_BOTTOM, 0, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
		}
	}

	void dynamic_worker(){
		int work_id;
		MPI_Status status;
		auto dx = (xmax-xmin)/chunks;

		if(mype < chunks){	//assigned for init work or not
			MPI_Recv(&work_id, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			while(status.MPI_TAG != 0){	//not a termination tag
				auto new_xmin = xmin + dx*work_id;
				auto new_xmax = (work_id != chunks -1)? (new_xmin + dx): xmax; //care about last one
				auto new_Nx = (work_id != chunks -1)? Nx/chunks: Nx/chunks+Nx%chunks;
				Julia j(new_xmin, new_xmax, ymin, ymax, new_Nx, Ny);
				vector<int> res = j.run();
				MPI_Send(res.data(), new_Nx*Ny, MPI_INT, 0, work_id+1, MPI_COMM_WORLD);
				MPI_Recv(&work_id, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			}
		}
	}

	void write_serial(std::string file, vector<int>& vec, int seek_pos){
		std::ofstream of(file, std::ios::in | std::ios::out | std::ios::binary);
		of.seekp(seek_pos, std::ios::beg);
		for(auto i=0; i<vec.size()/Ny; ++i)
			for(auto j=0; j<Ny; ++j)
				of.write(reinterpret_cast<char *>(&vec[i*Ny+j]), sizeof(int));

		of.close();
	}

	void write_mpi(std::string file, vector<int>& vec, int seek_pos){
		MPI_File fh;
		MPI_Offset offset;
		MPI_File_open(MPI_COMM_WORLD, const_cast<char*>(file.c_str()),
			MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

		for(auto i=0; i<vec.size()/Ny; ++i){
			for(auto j=0; j<Ny; ++j){
				offset = seek_pos + sizeof(int)*(i*Ny+j);
				MPI_File_seek(fh, offset, MPI_SEEK_SET);
				MPI_File_write(fh, &vec[i*Ny+j], 1, MPI_INT, MPI_STATUS_IGNORE);
			}
		}

		MPI_File_close(&fh);
	}
};






