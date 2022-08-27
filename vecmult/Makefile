SHELL = bash

.PHONY:
build: main.c vecmult.cl ../lib/opencl.c ../lib/opencl.h
	mkdir -p build/
	gcc main.c ../lib/opencl.c -o build/vecmult -l OpenCL
