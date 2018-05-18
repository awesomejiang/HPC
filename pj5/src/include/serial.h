#ifndef SERIAL_H
#define SERIAL_H

#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>

#include "vec3.h"


class Serial{
public:
	using Vec = Vec3<double>;

	Serial(Vec const &W, Vec const &C, Vec const &L, double rad);

	std::vector<double> &render(int n);
	void write(std::string filename);

private:
	Vec random_direction();
	double has_root(Vec const &v);

	Vec W, C, L;
	double rad;
	std::vector<double> G;
	std::random_device r;
	std::default_random_engine e_theta, e_phi;
	std::uniform_real_distribution<double> dis_phi;
	std::uniform_real_distribution<double> dis_cos_theta;
};

#endif