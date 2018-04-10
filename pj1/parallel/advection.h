#ifndef ADVECTION_H
#define ADVECTION_H

#include <omp.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <fstream>

using std::vector;
using std::string;

using matrix = vector<vector<double>>;

namespace advection{
class Advection{
public:
	Advection(int N, int NT, double L, double T, double u, double v)
		: N{N}, NT{NT}, L{L}, T{T}, u{u}, v{v},
		delta_x{L/N}, delta_t{T/NT},
		curr_mx{matrix(N, vector<double>(N, 0.0))},
		next_mx{matrix(N, vector<double>(N, 0.0))}
	{
		int size = delta_x/(std::sqrt(2)*std::hypot(u, v));
		//check Courant stability condition
		try{
			if(delta_t > delta_x/(std::sqrt(2)*std::hypot(u, v))){
				throw std::runtime_error("Args are instable.");
			}
		}
		catch (std::exception &e){
			std::cout << "Exception:\t" << e.what() << std::endl;
		}
	}
	//sig_x and sig_y should be hard coded in driver.
	void init_gaussian(double sig_x, double sig_y){
		#pragma omp parallel for default(none) shared(sig_x, sig_y)
		for(auto i=0; i<N; ++i){
			if(i==0){
				printf("num_thread:%d\n", omp_get_num_threads());
			}
			for(auto j=0; j<N; ++j){
				curr_mx[i][j] = 
					std::exp(-0.5*(pow((i*delta_x-L/2)/sig_x, 2) + pow((j*delta_x-L/2)/sig_y, 2)));
			}
		}
	}
	//run simulation and store results every 1000 timesteps
	void run(string file){
		for(auto t=0; t<NT; ++t){
			#pragma omp parallel for default(none)
			for(auto i=0; i<N; ++i){
				for(auto j=0; j<N; ++j){
					next_mx[i][j] = update_value(i, j);
				}
			}
			//write
			/*
			if(t%(NT/20) == 0){
				std::cout << "writing " << t/(NT/20) + 1 <<"th frame" << std::endl;
				for(auto vec: curr_mx){
					std::copy(vec.begin(), vec.end(),
						std::ostream_iterator<double>(of, "\t"));
					of << std::endl;
				}
			}
			*/
			std::swap(curr_mx, next_mx);
		}
	}

private:
	int N, NT;
	double L, T, u, v, delta_x, delta_t;
	matrix curr_mx, next_mx;

	double update_value(int i, int j){
		//periodic conditions
		double up = curr_mx[(i-1+N)%N][j],
			down = curr_mx[(i+1)%N][j],
			left = curr_mx[i][(j-1+N)%N],
			right = curr_mx[i][(j+1)%N];
		//"eq 6"
		double res = 0;
		return 0.25*(up+down+left+right)
			- delta_t/(2*delta_x)*(u*(down-up)+v*(right-left));
	}
};
}

#endif