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
	echo "Single core:"
	perf stat -n -- taskset -c 0 ./knn_cpu $(TRAINFILE) $(TESTFILE) OUTFILE 3
	echo "Multi core:"
	perf stat -n -- ./knn_cpu $(TRAINFILE) $(TESTFILE) OUTFILE 3
	echo "GPU"
	perf stat -n -- ./knn $(TRAINFILE) $(TESTFILE) OUTFILE 3

clean: knn knn_cpu
	rm -f $^
