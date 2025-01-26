LIBS=`pkg-config --libs OpenCL`

all: knn

knn: knn.c
	gcc -O2 $(LIBS) -o $@ $^
