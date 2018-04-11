all: parallel

parallel:	parallel.cpp advection.h
	g++ -fopenmp parallel.cpp -o parallel -std=c++14

clean:
	rm -f parallel