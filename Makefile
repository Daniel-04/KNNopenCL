LIBS=`pkg-config --libs OpenCL`

all: knn knn_cpu

knn: knn.c
	gcc -O2 $(LIBS) -o $@ $^

knn_cpu: knn.c
	gcc -O2 $(LIBS) -DDEVICE_TYPE=CL_DEVICE_TYPE_CPU -o $@ $^

bench: all
ifndef TRAINFILE
	$(error TRAINFILE must be declared, e.g., make bench TRAINFILE=train.csv)
endif
ifndef TESTFILE
	$(error TESTFILE must be declared, e.g., make bench TESTFILE=test.csv)
endif
	taskset -c 0    ./knn_cpu $(TRAINFILE) $(TESTFILE) OUTFILE 3
	taskset -c 0-1  ./knn_cpu $(TRAINFILE) $(TESTFILE) OUTFILE 3
	taskset -c 0-3  ./knn_cpu $(TRAINFILE) $(TESTFILE) OUTFILE 3
	taskset -c 0-7  ./knn_cpu $(TRAINFILE) $(TESTFILE) OUTFILE 3
	./knn $(TRAINFILE) $(TESTFILE) OUTFILE 3

clean: knn knn_cpu
	rm -f $^
