CC=nvcc
CFLAGS=-g -std=c++11 -O3#-DPTH

OBJS = $(patsubst include/%.h, obj/%.o, $(wildcard include/*.h))

all : dir main

$(OBJS): obj/%.o : include/%.cu include/%.h
	$(CC) $(CFLAGS) -dc -Iinclude -o $@ $<

main: obj/main.o $(OBJS)
	$(CC) $(CFLAGS) -Iinclude $< $(OBJS) -o $@

obj/%.o: %.cu
	$(CC) $(CFLAGS) -dc -Iinclude $< -o $@

dir:
	mkdir -p obj

clean :
	rm -rf main obj *.out