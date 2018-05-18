#include <iostream>
#include <cstdlib>
#include <chrono>
#include "vec3.h"
#include "serial.h"

using namespace std;

void run_serial(int n){
	cout << "Running Serial Ray Tracing..." << endl;

	auto t1 = chrono::steady_clock::now();

	// solve
	Serial s({10,10,10}, {0,12,0}, {4,4,-1}, 6);
	s.render(n);

	auto t2 = chrono::steady_clock::now();		
	std::cout << "Serial Running time: "
		 << chrono::duration_cast<chrono::milliseconds>(t2-t1).count()
		 << " ms" << endl;

	//write file
	s.write("serial.out");

}

int main(int argc, char **argv){
	run_serial(atoi(argv[1]));

	return 0;
}
