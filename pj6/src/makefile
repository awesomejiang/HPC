CC=g++
CFLAGS=-std=gnu++98#-DPTH

LIBS = $(patsubst include/%.h, lib/lib%.so, $(wildcard include/*.h))
TESTS = $(patsubst %.cpp, %, $(wildcard *.cpp))


all : $(TESTS)

$(LIBS): lib/lib%.so : include/%.cpp dir
	$(CC) $(CFLAGS) -fPIC -shared -Iinclude -o $@ $<

$(TESTS): %: obj/%.o $(LIBS)
	$(CC) $(CFLAGS) -Wl,-rpath lib -Iinclude $< -o $@ -Llib -lnbody -lvec3
	
obj/%.o: %.cpp dir
	$(CC) $(CFLAGS) -c -Iinclude -o $@ $<

dir:
	mkdir -p lib obj

clean :
	rm -rf $(TESTS) lib obj