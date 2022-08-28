SHELL = bash

build/vecmult: main.c vecmult.cl ../lib/opencl.h ../build/lib/libopencl.so
	mkdir -p build/
	gcc -Wall -Werror -I"../lib" -L"../build/lib" -Wl,-rpath="../build/lib" main.c -o build/vecmult -lopencl -lOpenCL
