SHELL = bash

build/lib/libopencl.so: lib/ lib/opencl.h lib/opencl.c
	mkdir -p ./build ./build/lib
	gcc -Wall -Werror -shared -fpic -o build/lib/libopencl.so lib/opencl.c

