HEADS = $(wildcard include/*.h)

all: main

main: $(HEADS) main.cpp
	mpic++ main.cpp -o main -I include -std=c++11

.PHONY:clean
clean:
	rm main *.out