#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>

#include "advection.h"

using namespace advection;


/**
	\brief common advection class.
**/
Advection_common::Advection_common(int N, int NT, double L, double T, double u, double v)
	: N{N}, NT{NT}, L{L}, T{T}, u{u}, v{v},
	delta_x{L/N}, delta_t{T/NT},
	curr_mx{matrix<double>(N, line<double>(N, 0.0))},
	next_mx{matrix<double>(N, line<double>(N, 0.0))} {}


double Advection_common::update_value(int i, int j){
	//periodic conditions
	double up = curr_mx[(i-1+N)%N][j],
		down = curr_mx[(i+1)%N][j],
		left = curr_mx[i][(j-1+N)%N],
		right = curr_mx[i][(j+1)%N];

	//"eq 6"
	return 0.25*(up+down+left+right)
		- delta_t/(2*delta_x)*(u*(down-up)+v*(right-left));
}