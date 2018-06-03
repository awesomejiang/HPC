#include "nbody.h"

typedef std::vector<Body>::iterator vbody_iter;

System::System(): n_bodies(100) {
	init_system();
}

System::System(int n_bodies): n_bodies(n_bodies) {
	init_system();
}

void System::init_system(){
	// Create MPI type
	#ifdef MPI_ON

	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &mype);
	n_bodies /= nprocs;

	// Build VEC type in MPI
	MPI_Datatype MPI_VEC;
	MPI_Datatype type_v[1] = {MPI_DOUBLE};
	int blocklen_v[1] = {3};
	MPI_Aint disp_v[1] = {0};
	MPI_Type_create_struct(1, blocklen_v, disp_v, type_v, &MPI_VEC);
	MPI_Type_commit(&MPI_VEC);

	// Build BODY type in MPI
	MPI_Datatype type_b[2] = {MPI_VEC, MPI_DOUBLE};
	int blocklen_b[2] = {2, 1};
	MPI_Aint disp_b[2] = {0, sizeof(Vec)*2};
	MPI_Type_create_struct(2, blocklen_b, disp_b, type_b, &MPI_BODY);
	MPI_Type_commit(&MPI_BODY);

	#endif

	// Init bodies
	init_bodies();
}

void System::init_bodies(){
	bodies.assign(n_bodies, Body());

	#ifdef MPI_ON

	MPI_Request r;
	if(mype == 0){
		galaxy(); // init bodies and save it temporarily.
		in_buffer = bodies;
		for(int rank=1; rank<nprocs; ++rank){			
			galaxy();
			if(mype != rank)
				MPI_Isend(&bodies.front(), n_bodies, MPI_BODY, rank, 0, MPI_COMM_WORLD, &r);
		}
		bodies = in_buffer; // restore values into bodies.
	}
	else
		MPI_Irecv(&bodies.front(), n_bodies, MPI_BODY, 0, 0, MPI_COMM_WORLD, &r);

	MPI_Wait(&r, MPI_STATUS_IGNORE);

	#else

	galaxy();

	#endif

}

#ifdef MPI_ON

void System::run_simulation(double dt, int n_iters, char *fname){
	// Open File
	MPI_File fh;
	MPI_Offset offset;
	MPI_File_open(MPI_COMM_SELF, fname, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

	// Write Header Info
	if(mype == 0){
		char tmp[512];
		sprintf(tmp, "%+.*le %+.*le %+.*le\n", 10, (double)(n_bodies*nprocs), 10, (double) n_iters, 10, 0.0);
		MPI_File_seek(fh, 0, MPI_SEEK_SET);
		MPI_File_write(fh, tmp, strlen(tmp), MPI_CHAR, MPI_STATUS_IGNORE);
	}

	double start = get_time();

	MPI_Request r;
	// Loop over timesteps
	for (int i = 0; i < n_iters; i++)
	{
		if(mype == 0)
			printf("iteration: %d\n", i);

		for(vbody_iter iter = bodies.begin(); iter != bodies.end(); ++iter){
			char tmp[512];
			sprintf(tmp, "%+.*le %+.*le %+.*le\n", 10, iter->r.x, 10, iter->r.y, 10, iter->r.z);
			offset = strlen(tmp)*(1+n_bodies*(nprocs*i+mype)+iter-bodies.begin());
			MPI_File_seek(fh, offset, MPI_SEEK_SET);
			MPI_File_write(fh, tmp, strlen(tmp), MPI_CHAR, MPI_STATUS_IGNORE);
		}

		// Compute new forces & velocities for all particles
		// Compute forces between bodies and in_buffers
		// then "rotate" buffer to neighbors
		in_buffer = bodies;
		for(int rank=0; rank<nprocs; ++rank){
			compute_forces(dt);
			out_buffer = in_buffer;
			MPI_Isend(&out_buffer.front(), n_bodies, MPI_BODY, (mype+nprocs-1)%nprocs, 0, MPI_COMM_WORLD, &r);
			MPI_Irecv(&in_buffer.front(), n_bodies, MPI_BODY, (mype+1)%nprocs, 0, MPI_COMM_WORLD, &r);
			MPI_Wait(&r, MPI_STATUS_IGNORE);
		}

		// Update positions of all particles
		for(vbody_iter iter = bodies.begin(); iter != bodies.end(); ++iter)
			iter->r += iter->v*dt;
	}

	// Close data file
	MPI_File_close(&fh);

	double stop = get_time();

	if(mype == 0){
		double runtime = stop-start;
		double time_per_iter = runtime / n_iters;
		long interactions = n_bodies * n_bodies;
		double interactions_per_sec = interactions / time_per_iter;

		printf("SIMULATION COMPLETE\n");
		printf("Runtime [s]:              %.3le\n", runtime);
		printf("Runtime per Timestep [s]: %.3le\n", time_per_iter);
		printf("Interactions per sec:     %.3le\n", interactions_per_sec);
	}
}

#else

void System::run_simulation(double dt, int n_iters, char *fname){
	// Open File and Write Header Info
	FILE * datafile = fopen(fname,"w");
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

#endif

void System::compute_forces(double dt){
	double G = 6.67259e-3;
	double softening = 1.0e-5;

	#ifdef OPENMP_ON

	#pragma omp parallel for

	#endif

	// For each particle in the set
	for(int i = 0; i < bodies.size(); ++i){
		vbody_iter iter_i = bodies.begin() + i;

		Vec F(0.0, 0.0, 0.0);

		// Compute force from all other particles in the set
		#ifdef MPI_ON

		for(vbody_iter iter_j = in_buffer.begin(); iter_j != in_buffer.end(); ++iter_j){
		
		#else

		for(vbody_iter iter_j = bodies.begin(); iter_j != bodies.end(); ++iter_j){

		#endif
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

void System::uniform_random(){
	static int flag = 0;

	//generate rng
	if(flag == 0)
		srand(42);

	//randomize bodies;
	double vm = 1.0e-3; // velocity scaling term

	for(int i=0; i<n_bodies; ++i){
		bodies[i] = Body(
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
					);
	}
	if(flag == 0)
		++flag;
}

// A interesting case, work for 3 bodies only
void System::three_body(){
	double G = 6.67259e-3;
	double m = 10;
	bodies.push_back(
		Body(Vec(0, sqrt(3)/3, 0),
			 Vec(sqrt(G*m), 0, 0),
			 m
		)
	);
	bodies.push_back(
		Body(Vec(0.5, -sqrt(3)/6, 0),
			 Vec(-sqrt(G*m)/2, -sqrt(3*G*m)/2, 0),
			 m
		)
	);
	bodies.push_back(
		Body(Vec(-0.5, -sqrt(3)/6, 0),
			 Vec(-sqrt(G*m)/2, sqrt(3*G*m)/2, 0),
			 m
		)
	);	
}

void System::shpere(){
	static int flag = 0;
	
	//generate rng
	if(flag == 0)
		srand(42);

	for(int i=0; i<n_bodies; ++i){
		double phi = double(rand()/(double)RAND_MAX)*2*M_PI;
		double sin_ph = sin(phi), cos_ph = cos(phi);

		double cos_th = (double(rand()/(double)RAND_MAX)-.5)*2;
		double sin_th = sqrt(1-cos_th*cos_th);

		
		bodies[i] = Body(
						Vec(sin_th*cos_ph, sin_th*sin_ph, cos_th),
						Vec(0, 0, 0),
						1.0 / n_bodies
					);
	}
	if(flag == 0)
		++flag;
}

void System::galaxy(){
	static int flag = 0;
	
	//generate rng
	if(flag == 0)
		srand(42);

	double G = 6.67259e-3, M = 1, m  = 10e-6;

	for(int i=0; i<n_bodies; ++i){
		double phi = double(rand()/(double)RAND_MAX)*2*M_PI;
		double radius = double(rand()/(double)RAND_MAX);
		double velocity = sqrt(G*M/radius);
		bodies[i] = Body(
						Vec(radius*cos(phi), radius*sin(phi), 0),
						Vec(velocity*sin(phi), -velocity*cos(phi), 0 ),
						m
					);
	}
	if(flag == 0){
	// set one and only one HUGE star at center.
		bodies[0] = Body(Vec(0, 0, 0), Vec(0, 0, 0), M);
		++flag;
	}
}

double get_time(){
	#ifdef MPI_ON
	return MPI_Wtime();
	#endif

	#ifdef OPENMP_ON
	return omp_get_wtime();
	#endif

	time_t time;
	time = clock();

	return (double) time / (double) CLOCKS_PER_SEC;
}
