#ifndef RENDER_H
#define RENDER_H

#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <cmath>
#include <cuda.h>
#include <string>

#include "vec3.h"

using Vec = Vec3<double>;

class Scene{
public:

	Scene(Vec const &W, Vec const &C, Vec const &L, double rad);

	Vec W, C, L;
	double rad;
};

void render_serial(Scene const &p, int ndim, double *G, int rays);
Vec random_serial();

void render_cuda(Scene const &p, int ndim, double *G, int blocks, int threads);
__device__ Vec random_cuda(curandState *state);
__global__ void render_thread(Scene *p, int ndim, double *G);
__device__ double atomicAdd_d(double* address, double val);

#endif