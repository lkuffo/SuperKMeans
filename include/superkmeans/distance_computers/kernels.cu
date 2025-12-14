#include <cstdio>

#include "superkmeans/distance_computers/kernels.cuh"

__global__ void test_kernel() {
	printf("Hello from the kernel!\n");
}

void test() {
	printf("Hello from the host!\n");
	test_kernel<<<1, 1>>>();
}
