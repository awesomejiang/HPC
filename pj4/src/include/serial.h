#ifndef SERIAL_H
#define SERIAL_H

#include "poisson.h"

class Serial: public Poisson{
public:
	Serial(long n);
	
	vec solve() override;

private:
	vec matvec(vec const &w) override;
	double dotp(vec const &a, vec const &b) override;
	vec axpy(double a, vec const &w, double b, vec const &v) override;
	vec fill_b() override;
};

#endif