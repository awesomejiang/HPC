#include "render.h"

using Vec = Vec3<double>;

Scene::Scene(Vec const &W, Vec const &C, Vec const &L, double rad)
: W{W}, C{C}, L{L}, rad{rad} {}


/* global functions */

void render_serial(Scene const &p, int ndim, double *G, int rays){
	for(auto i=0; i<rays; ++i){
		Vec P, V, I;
		double root, t;
		bool flag = true;
		while(flag){
			V = random_serial();
			P = (p.W.y/V.y)*V;
			if((-p.W.x<P.x || P.x>p.W.x) && (-p.W.z<P.z || P.z>p.W.z)){	
				//if has root
				root = V*p.C;
				root *= root;
				root += p.rad*p.rad - p.C*p.C;
				if(root>0)
					flag = false;
			}
		}
		t = V*p.C - sqrt(root);
		I = t*V;

		int g_x = (P.x+p.W.x)/(p.W.x*2)*ndim,
			g_z = (P.z+p.W.z)/(p.W.z*2)*ndim;
		G[g_x*ndim + g_z] += max(0.0, (I-p.C).norm()*(p.L-I).norm());
	}
}

Vec random_serial(){
	static std::random_device r;
	static std::default_random_engine e_theta(r()), e_phi(r());
	static std::uniform_real_distribution<double> dis_phi(0, 2*M_PI),
												  dis_cos_theta(-1, 1);

	auto phi = dis_phi(e_phi);
	auto sin_ph = std::sin(phi), cos_ph = std::cos(phi);
	auto cos_th = dis_cos_theta(e_theta);
	auto sin_th = std::sqrt(1-cos_th*cos_th);

	return Vec{sin_th*cos_ph, sin_th*sin_ph, cos_th};
}


void print_error(){
	std::string error = cudaGetErrorString(cudaPeekAtLastError());
	printf("1:%s\n", error.c_str());
	error = cudaGetErrorString(cudaThreadSynchronize());
	printf("2:%s\n", error.c_str());
}

void render_cuda(Scene const &p, int ndim, double *G, dim3 grids, dim3 blocks){
	Scene *dev_p;
	double *dev_G;
	cudaMalloc( (void **) &dev_p, sizeof(Scene));
	cudaMalloc( (void **) &dev_G, sizeof(double)*ndim*ndim);

	cudaMemcpy(dev_p, &p, sizeof(Scene), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_G, G, sizeof(double)*ndim*ndim, cudaMemcpyHostToDevice);

  	render_thread<<<grids, blocks>>>(dev_p, ndim, dev_G);
	print_error();

	cudaMemcpy(G, dev_G, sizeof(double)*ndim*ndim, cudaMemcpyDeviceToHost);

	cudaFree(dev_p);
	cudaFree(dev_G);
}

__device__ Vec random_cuda(curandState *state){
	auto phi = curand_uniform_double(state)*2*M_PI;
	auto sin_ph = std::sin(phi), cos_ph = std::cos(phi);

	auto cos_th = (curand_uniform_double(state)-.5)*2;
	auto sin_th = std::sqrt(1-cos_th*cos_th);

	return Vec{sin_th*cos_ph, sin_th*sin_ph, cos_th};
}

__global__ void render_thread(Scene *p, int ndim, double *G){
	//int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idx = (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x + threadIdx.x;

	curandState state;
	curand_init(clock64()+idx, 0, 0, &state);

	Vec V, P;
	double root;
	bool flag = true;
	while(flag){
		V = random_cuda(&state);
		P = (p->W.y/V.y)*V;
		if((-p->W.x<P.x || P.x>p->W.x) && (-p->W.z<P.z || P.z>p->W.z)){	
			//if has root
			root = V*p->C;
			root *= root;
			root += p->rad*p->rad - p->C*p->C;
			if(root>0)
				flag = false;
		}
	}

	double t = V*p->C - sqrt(root);
	Vec I = t*V;

	int g_x = (P.x+p->W.x)/(p->W.x*2)*ndim,
		g_z = (P.z+p->W.z)/(p->W.z*2)*ndim;

	atomicAdd_d(G + g_x*ndim + g_z, max(0.0, (I-p->C).norm()*(p->L-I).norm()));
}

__device__ double atomicAdd_d(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}