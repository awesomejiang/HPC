#include <algorithm>
#include "poisson.h"

Poisson::Poisson(long n): n{n} {}

double Poisson::find_b(long i, long j){
	double delta = 1.0 / n;

	double x = -.5 + delta + delta * j, y = -.5 + delta + delta * i;

	// Check if within a circle, radius = 0.1
	return (x*x + y*y < 0.01)?delta * delta / 1.075271758e-02 : 0.0;
}