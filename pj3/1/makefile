SRCS = $(wildcard *.cpp)
OBJS = $(patsubst %.cpp, %, $(SRCS))

all: $(OBJS)

$(OBJS): %: %.cpp
	mpic++ $< -o $@ -std=c++11

.PHONY:clean
clean:
	rm $(OBJS)