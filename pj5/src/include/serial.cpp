#include "serial.h"

using Vec = Vec3<double>;

Serial::Serial(Vec const &W, Vec const &C, Vec const &L, double rad)
: W{W}, C{C}, L{L}, e_theta(r()), e_phi(r()), rad{rad},
  dis_phi(0, 2*M_PI), dis_cos_theta(-1, 1) {}


Vec Serial::random_direction(){
	auto phi = dis_phi(e_phi);
	auto sin_ph = std::sin(phi), cos_ph = std::cos(phi);
	auto cos_th = dis_cos_theta(e_theta);
	auto sin_th = std::sqrt(1-cos_th*cos_th);

	return Vec{sin_th*cos_ph, sin_th*sin_ph, cos_th};
}

double Serial::has_root(Vec const &V){
	auto res = V*C;
	res *= res;
	return res + rad*rad - C*C;
}

std::vector<double> &Serial::render(int n){
	G = std::vector<double>(n*n);

	for(auto i=0; i<n*n*100; ++i){
		Vec P, V, I;
		double root, t, b;
		bool flag = true;

		while(flag){
			V = random_direction();
			P = (W.y/V.y)*V;
			root = has_root(V);
			if((-W.x<P.x || P.x>W.x) && (-W.z<P.z || P.z>W.z)
				&& root>0)
				flag = false;
		}

		t = V*C - std::sqrt(root);
		I = t*V;

		int g_x = (P.x+W.x)/(W.x*2)*n, g_z = (P.z+W.z)/(W.z*2)*n;
		G[g_x*n + g_z] += std::max(0.0, (I-C).norm()*(L-I).norm());
	}
	return G;
}

void Serial::write(std::string filename){
	std::ofstream of(filename,
		std::ios::trunc | std::ios::in | std::ios::out | std::ios::binary);
	of.write(reinterpret_cast<char *>(G.data()), G.size()*sizeof(double));
	of.close();
}