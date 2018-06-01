#include "nbody.h"

using std::rand;
using std::printf;
using std::fprintf;
typedef std::vector<Body>::iterator vbody_iter;

Body::Body(): r(0, 0, 0), v(0, 0, 0), m(0) {}

Body::Body(Vec r, Vec v, double m)
: r(r), v(v), m(m) {}

System::System(): n_bodies(100) {
	init_system();
}

System::System(int n_bodies): n_bodies(n_bodies) {
	init_system();
}

void System::init_system(){
/*
	uniform random:	
	//generate rng
	std::srand(42);

	//randomize bodies;
	double vm = 1.0e-3; // velocity scaling term

	for(int i=0; i<n_bodies; ++i){
		bodies.push_back(
			Body(
				Vec(
					2.0 * (rand() / (double)RAND_MAX) - 1.0,
					2.0 * (rand() / (double)RAND_MAX) - 1.0,
					2.0 * (rand() / (double)RAND_MAX) - 1.0
				),
				Vec(
					2.0*vm * (rand() / (double)RAND_MAX) - vm,
					2.0*vm * (rand() / (double)RAND_MAX) - vm,
					2.0*vm * (rand() / (double)RAND_MAX) - vm
				),
				1.0 / n_bodies
			)
		);
	}

	3-body:
	double G = 6.67259e-3;
	double m = 10;
	bodies.push_back(
		Body{{0, std::sqrt(3)/3, 0}, {std::sqrt(G*m), 0, 0}, m}
	);
	bodies.push_back(
		Body{{0.5, -std::sqrt(3)/6, 0}, {-std::sqrt(G*m)/2, -std::sqrt(3*G*m)/2, 0}, m}
	);
	bodies.push_back(
		Body{{-0.5, -std::sqrt(3)/6, 0}, {-std::sqrt(G*m)/2, std::sqrt(3*G*m)/2, 0}, m}
	);

	shpere:
	//generate rng
	std::srand(42);

	for(int i=0; i<n_bodies; ++i){
		double phi = double(rand()/(double)RAND_MAX)*2*M_PI;
		double sin_ph = std::sin(phi), cos_ph = std::cos(phi);

		double cos_th = (double(rand()/(double)RAND_MAX)-.5)*2;
		double sin_th = std::sqrt(1-cos_th*cos_th);

		
		bodies.push_back(
			Body(
				Vec(sin_th*cos_ph, sin_th*sin_ph, cos_th),
				Vec(0, 0, 0),
				1.0 / n_bodies
			)
		);
	}

	galaxy:
	//generate rng
	std::srand(42);

	double G = 6.67259e-3;
	double M = 1;
	double m  = 10e-6;

	bodies.push_back(
		Body(
			Vec(0, 0, 0),
			Vec(0, 0, 0),
			M
		)
	);

	for(int i=1; i<n_bodies; ++i){
		double phi = double(rand()/(double)RAND_MAX)*2*M_PI;
		double radius = double(rand()/(double)RAND_MAX);
		double velocity = std::sqrt(G*M/radius);
		bodies.push_back(
			Body(
				Vec(
					radius*std::cos(phi),
					radius*std::sin(phi),
					0
				),
				Vec(
					velocity*std::sin(phi),
					-velocity*std::cos(phi),
					0
				),
				m
			)
		);
	}
*/	
//generate rng
	std::srand(42);

	double G = 6.67259e-3;
	double M = 0.1;
	double m  = 10e-6;

	bodies.push_back(
		Body(
			Vec(0, 0, 0),
			Vec(0, 0, 0),
			M
		)
	);

	for(int i=1; i<n_bodies; ++i){
		double phi = double(rand()/(double)RAND_MAX)*2*M_PI;
		double radius;
		if(i<0.75*n_bodies)
			radius = phi/10*(1+double(rand()/(double)RAND_MAX)*0.2);
		else
			radius = double(rand()/(double)RAND_MAX);
		double velocity = std::sqrt(G*M/radius);

		bodies.push_back(
			Body(
				Vec(
					radius*std::cos(phi),
					radius*std::sin(phi),
					0
				),
				Vec(
					velocity*std::sin(phi),
					-velocity*std::cos(phi),
					0
				),
				m
			)
		);
	}
}

void System::run_simulation(double dt, int n_iters, char *fname){
	// Open File and Write Header Info
	FILE * datafile = fopen("nbody.dat","w");
	fprintf(datafile, "%+.*le %+.*le %+.*le\n", 10, (double)n_bodies, 10, (double) n_iters, 10, 0.0);

	double start = get_time();

	// Loop over timesteps
	for (int i = 0; i < n_iters; i++)
	{
		printf("iteration: %d\n", i);

		// Output body positions to file
		for(vbody_iter iter = bodies.begin(); iter != bodies.end(); ++iter)
			fprintf(datafile, "%+.*le %+.*le %+.*le\n", 10, iter->r.x, 10, iter->r.y, 10, iter->r.z);

		// Compute new forces & velocities for all particles
		compute_forces(dt);

		// Update positions of all particles
		for(vbody_iter iter = bodies.begin(); iter != bodies.end(); ++iter)
			iter->r += iter->v*dt;

	}

	// Close data file
	fclose(datafile);

	double stop = get_time();

	double runtime = stop-start;
	double time_per_iter = runtime / n_iters;
	long interactions = n_bodies * n_bodies;
	double interactions_per_sec = interactions / time_per_iter;

	printf("SIMULATION COMPLETE\n");
	printf("Runtime [s]:              %.3le\n", runtime);
	printf("Runtime per Timestep [s]: %.3le\n", time_per_iter);
	printf("Interactions per sec:     %.3le\n", interactions_per_sec);
}

void System::compute_forces(double dt){
	double G = 6.67259e-3;
	double softening = 1.0e-5;

	// For each particle in the set
	for(vbody_iter iter_i = bodies.begin(); iter_i != bodies.end(); ++iter_i){
		Vec F(0.0, 0.0, 0.0);

		// Compute force from all other particles in the set
		for(vbody_iter iter_j = bodies.begin(); iter_j != bodies.end(); ++iter_j){
			// F_ij = G * [ (m_i * m_j) / distance^3 ] * (location_j - location_i) 

			// First, compute the "location_j - location_i" values for each dimension
			Vec dr = iter_j->r - iter_i->r;

			// Then, compute the distance^3 value
			// We will also include a "softening" term to prevent near infinite forces
			// for particles that come very close to each other (helps with stability)

			// distance = sqrt( dx^2 + dx^2 + dz^2 )
			double distance = sqrt(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z + softening);
			double distance_cubed = distance * distance * distance;

			// Now compute G * m_2 * 1/distance^3 term, as we will be using this
			// term once for each dimension
			// NOTE: we do not include m_1 here, as when we compute the change in velocity
			// of particle 1 later, we would be dividing this out again, so just leave it out
			double m_j = iter_j->m;
			double mGd = G * m_j / distance_cubed;
			F += mGd * dr;
		}

		// With the total forces on particle "i" known from this batch, we can then update its velocity
		// v = (F * t) / m_i
		// NOTE: as discussed above, we have left out m_1 from previous velocity computation,
		// so this is left out here as well
		iter_i->v += dt*F;
	}
}

double get_time(){
	#ifdef MPI
	return MPI_Wtime();
	#endif

	#ifdef OPENMP
	return omp_get_wtime();
	#endif

	time_t time;
	time = std::clock();

	return (double) time / (double) CLOCKS_PER_SEC;
}