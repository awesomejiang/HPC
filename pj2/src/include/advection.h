#ifndef ADVECTION
#define ADVECTION

#include <vector>
#include <string>

template<typename T>
using line = std::vector<T>;

template<typename T>
using matrix = std::vector<line<T>>;

namespace advection{

class Advection_common{
public:
	Advection_common(int N, int NT, double L, double T, double u, double v);

	int N, NT;
	double L, T, u, v, delta_x, delta_t;
	matrix<double> curr_mx, next_mx;

	//x0, y0 is left-up corner of matrix
	virtual void init_gaussian(double sig_x, double sig_y, double x0, double y0) = 0;
	virtual void run() = 0;
	virtual void run(std::string file) = 0;
	double update_value(int i, int j);
};

class Advection_serial: public Advection_common {
public: 
	Advection_serial(int N, int NT, double L, double T, double u, double v);
	void init_gaussian(double sig_x, double sig_y, double x0, double y0) override;
	void run() override;
	void run(std::string file) override;
};

class Advection_threads: public Advection_common {
public: 
	Advection_threads(int N, int NT, double L, double T, double u, double v);
	void init_gaussian(double sig_x, double sig_y, double x0, double y0) override;
	void run() override;
	void run(std::string file) override;

};

class Advection_mpi_blocking: public Advection_common {
public: 
	Advection_mpi_blocking(int N, int NT, double L, double T, double u, double v, int mype, int k);
	void init_gaussian(double sig_x, double sig_y, double x0, double y0) override;
	void run() override;
	void run(std::string file) override;

	int mype, k;
	std::vector<std::vector<double>> ghost_cells; //0,1,2,3->up, right, down, left
	double update_value(int i, int j);
	void sync();

private:
	void sync_up();
	void sync_down();
	void sync_left();
	void sync_right();
};

class Advection_mpi_non_blocking: public Advection_mpi_blocking {
public: 
	Advection_mpi_non_blocking(int N, int NT, double L, double T, double u, double v, int mype, int k);
	void sync();
};

class Advection_hybrid: public Advection_mpi_blocking {
public: 
	Advection_hybrid(int N, int NT, double L, double T, double u, double v, int mype, int k);
	void init_gaussian(double sig_x, double sig_y, double x0, double y0) override;
	void run() override;
	void run(std::string file) override;
};

class Driver{
public:
	Driver(int argc, char**argv);

	void run();

private:
	char **argv;
	int argc, N, NT, thread;
	double L, T, u, v;
	std::string mode;

	void serial();
	void threads();
	void mpi_blocking();
	void mpi_non_blocking();
	void hybrid();
};

}

#endif