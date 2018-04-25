SRCS = $(patsubst include/%.cpp, %, $(wildcard include/*.cpp))
OBJS = $(patsubst %, %.o, $(SRCS))


CXX = mpic++ -fopenmp -std=c++14 -O3

all: $(SRCS) main
	
$(SRCS):  include/advection.h
	$(CXX) -c include/$@.cpp

main:	main.cpp $(OBJS)
	$(CXX) -fopenmp main.cpp  $(OBJS) -o main

clean:
	rm -f main *.o *.out *.gif