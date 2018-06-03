#=========================================================
# User Options
#=========================================================
COMPILER = gnu
MPI		 = yes
OPENMP	 = yes
OPTIMIZE = yes

#=========================================================
# Set flags
#=========================================================
LIBS = $(patsubst include/%.h, lib/lib%.so, $(wildcard include/*.h))
MAIN = $(patsubst %.cpp, %, $(wildcard *.cpp))

# Regular g++ compiler
ifeq ($(COMPILER),gnu)
  CC = g++
endif

CFLAGS=-std=gnu++98 -g

# MPI compiler
ifeq ($(MPI),yes)
  CC = mpicxx
  CFLAGS += -DMPI_ON
endif

# OpenMP
ifeq ($(OPENMP),yes)
  CFLAGS += -fopenmp -DOPENMP_ON
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif


#=========================================================
# Building rules
#=========================================================
all : $(MAIN)

$(LIBS): lib/lib%.so : include/%.cpp dir
	$(CC) $(CFLAGS) -fPIC -shared -Iinclude -o $@ $<

$(MAIN): %: obj/%.o $(LIBS)
	$(CC) $(CFLAGS) -Wl,-rpath lib -Iinclude $< -o $@ -Llib -lnbody -lvec3
	
obj/%.o: %.cpp dir
	$(CC) $(CFLAGS) -c -Iinclude -o $@ $<

dir:
	mkdir -p lib obj dat

clean :
	rm -rf $(MAIN) lib obj
