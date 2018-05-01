#ifndef JULIA__H
#define JULIA__H

#include <string>
#include <vector>

using std::vector;

class Julia{
public:
	Julia(double xmin, double xmax, double ymin, double ymax, int Nx, int Ny)
		: xmin{xmin}, xmax{xmax}, ymin{ymin}, ymax{ymax},
		dx{(xmax-xmin)/Nx}, dy{(ymax-ymin)/Ny}, Nx{Nx}, Ny{Ny},
		values{vector<int>(Nx*Ny, 0)} {}

	vector<int> &run(){
		for(auto i=0; i<Nx; ++i)
			for(auto j=0; j<Ny; ++j)
				values[i*Ny+j] = iteration(i, j);
		
		return values;
	}

	int iteration(int i, int j){
		auto re = xmin + dx*i;
		auto im = ymin + dy*j;
		auto ctr = 0;
		while(re*re+im*im < 4.0 && ctr <= 1000){
			auto tmp = re*re - im*im;
			im = 2*re*im + 0.26;
			re = tmp - 0.7;
			++ctr;
		}
		return ctr;
	}

	int Nx, Ny;
	double xmin, xmax, ymin, ymax, dx, dy;
	vector<int> values;

};

#endif