#include "render.h"

using Vec = Vec3<double>;

Scene::Scene(Vec const &W, Vec const &C, Vec const &L, double rad)
: W{W}, C{C}, L{L}, rad{rad} {}


/* global functions */


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


__device__ Vec random_cuda(){
	static curandState state;
	static int flag = 0;
	if(flag++ == 0){
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		curand_init(clock64(), idx, 0, &state);
	}

	auto phi = curand_uniform_double(&state)*2*M_PI;
	auto sin_ph = std::sin(phi), cos_ph = std::cos(phi);

	auto cos_th = (curand_uniform_double(&state)-.5)*2;
	auto sin_th = std::sqrt(1-cos_th*cos_th);

	return Vec{sin_th*cos_ph, sin_th*sin_ph, cos_th};
}


__global__ void render_cuda(Scene p, int ndim, double *G){
	Vec P, V, I;
	double root, t;
	bool flag = true;
	while(flag){
		V = random_cuda();
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
