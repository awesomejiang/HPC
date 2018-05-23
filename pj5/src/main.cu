#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include "render.h"

using namespace std;

#define MAX_BLOCKS_PER_DIM 65535


void write(string filename, double* G, int len){
	ofstream of(filename,
		ios::trunc | ios::in | ios::out | ios::binary);
	of.write(reinterpret_cast<char *>(G), len*sizeof(double));
	of.close();
}


void run(int argc, char** argv){
	//read common args
	string mode = string{argv[1]};
	int ndim = atoi(argv[2]);
	if(mode != "serial" && mode != "cuda"){
		cout << "Unrecognized mode!" << endl;
		return ;
	}
	
	//int blocks, threads_per_block;
	//blocks = MIN(rays/threads_per_block + 1, MAX_BLOCKS_PER_DIM);

	//rendering...
	cout << "Running " << mode << " Ray Tracing..." << endl;

	Scene p({10,10,10}, {0,12,0}, {4,4,-1}, 6);
	double *G = (double*)malloc(sizeof(double)*ndim*ndim);
	for(auto i=0; i<ndim*ndim; ++i)
		G[i] = 0;

	cudaEvent_t start_device, stop_device;
	float time_device;
	cudaEventCreate(&start_device);
	cudaEventCreate(&stop_device);

	cudaEventRecord(start_device, 0);

	if(mode == "serial"){
		int rays = atoi(argv[3]);
  		render_serial(p, ndim, G, rays);
	}
	else if(mode == "cuda"){
		int b = atoi(argv[3]),
			th = atoi(argv[4]);

		Scene *dev_p;
		double *dev_G;
		cudaMalloc( (void **) &dev_p, sizeof(Scene));
		cudaMalloc( (void **) &dev_G, sizeof(double)*ndim*ndim);
		cudaMemcpy(dev_p, &p, sizeof(Scene), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_G, G, sizeof(double)*ndim*ndim, cudaMemcpyHostToDevice);

	  	render_cuda<<<b, th>>>(dev_p, ndim, dev_G);

		cudaMemcpy(G, dev_G, sizeof(double)*ndim*ndim, cudaMemcpyDeviceToHost);
		cudaFree(dev_p); 
		cudaFree(dev_G); 
	}

	cudaEventRecord(stop_device, 0);
	cudaEventSynchronize(stop_device);
	cudaEventElapsedTime(&time_device, start_device, stop_device);
	cout << "Running time: "
		 << time_device << " ms" << endl;

	cudaEventDestroy(start_device);
	cudaEventDestroy(stop_device);

	//write file
	write(mode+".out", G, ndim*ndim);

	free(G);
}


int main(int argc, char **argv){

	run(argc, argv);

	return 0;
}