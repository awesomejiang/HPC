#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iterator>

#include "advection.h"

using namespace advection;

Advection_serial::Advection_serial(int N, int NT, double L, double T, double u, double v)
	: Advection_common(N, NT, L, T, u, v){}

void Advection_serial::init_gaussian(double sig_x, double sig_y, double x0, double y0){
	for(auto i=0; i<N; ++i){
		for(auto j=0; j<N; ++j){
			curr_mx[i][j] = 
				std::exp(-0.5*(pow((i*delta_x-x0)/sig_x, 2) + pow((j*delta_x-y0)/sig_y, 2)));
		}
	}
}

void Advection_serial::run(){
	for(auto t=0; t<NT; ++t){
		for(auto i=0; i<N; ++i){
			for(auto j=0; j<N; ++j){
				next_mx[i][j] = update_value(i, j);
			}
		}
		std::swap(curr_mx, next_mx);
	}
}
void Advection_serial::run(std::string file){
	std::ofstream of(file, std::ios::out | std::ios::binary);
	for(auto t=0; t<NT; ++t){
		for(auto i=0; i<N; ++i){
			for(auto j=0; j<N; ++j){
				next_mx[i][j] = update_value(i, j);
			}
		}
		//write
		if(t%(NT/20) == 0){
			for(auto vec: curr_mx)
				of.write(reinterpret_cast<char*>(vec.data()), std::streamsize(N*sizeof(double)));
		}
		std::swap(curr_mx, next_mx);
	}
	of.close();
}