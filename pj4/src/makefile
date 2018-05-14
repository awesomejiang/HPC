CC=mpic++
CFLAGS=-g -std=c++11 #-DPTH

OBJS = $(patsubst include/%.h, obj/%.o, $(wildcard include/*.h))

all : dir main

$(OBJS): obj/%.o : include/%.cpp
	$(CC) $(CFLAGS) -c -Iinclude -o $@ $<

main: obj/main.o $(OBJS)
	$(CC) $(CFLAGS) -Iinclude $< $(OBJS) -o $@

obj/%.o: %.cpp
	$(CC) $(CFLAGS) -c -Iinclude $< -o $@

dir:
	mkdir -p obj

clean :
	rm -rf main obj *.out