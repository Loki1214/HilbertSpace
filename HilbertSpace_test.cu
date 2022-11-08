#include <catch2/catch_test_macros.hpp>
#include "HilbertSpace.hpp"
#include <random>
#include <iostream>
#include <cassert>

#define cuCHECK(call)                                                          \
	{                                                                          \
		const cudaError_t error = call;                                        \
		if(error != cudaSuccess) {                                             \
			printf("cuCHECK Error: %s:%d,  ", __FILE__, __LINE__);             \
			printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
			assert(error == cudaSuccess);                                      \
		}                                                                      \
	};

__global__ void test_HilbertSpace_kernel(Size dim) {
	{
		// Default constructor
		{
			HilbertSpace<int> hSpace;
			assert(hSpace.dim() == 0);
		}
		HilbertSpace<int> hSpace(dim);
		assert(hSpace.dim() == dim);
	}
	{
		// Copy constructor
		HilbertSpace<int> hSpace1(dim);
		HilbertSpace<int> hSpace2(hSpace1);
		assert(hSpace1.dim() == hSpace2.dim());
	}
	{
		// Move constructor
		HilbertSpace<int> hSpace1(dim);
		HilbertSpace<int> hSpace2(std::move(hSpace1));
		assert(hSpace2.dim() == dim);
	}

	{
		// Copy assignment operator
		HilbertSpace<int> hSpace1(dim);
		HilbertSpace<int> hSpace2;
		hSpace2 = hSpace1;
		assert(hSpace1.dim() == hSpace2.dim());
	}
	{
		// Move assignment operator
		HilbertSpace<int> hSpace1(dim);
		HilbertSpace<int> hSpace2;
		hSpace2 = std::move(hSpace1);
		assert(hSpace2.dim() == dim);
	}
	{
		// Equality operator
		HilbertSpace<int> hSpace1(dim);
		HilbertSpace<int> hSpace2(hSpace1);
		assert(hSpace1.dim() == hSpace2.dim());
		assert(hSpace1 == hSpace2);
	}
}

TEST_CASE("HilbertSpace_onGPU", "test") {
	std::random_device              seed_gen;
	std::default_random_engine      engine(seed_gen());
	std::uniform_int_distribution<> dist(0, 100000);

	Size testLoop = 100;

	for(Size n = 0; n != testLoop; ++n) {
		Size dim = dist(engine);
		test_HilbertSpace_kernel<<<1, 1>>>(dim);
		cuCHECK(cudaGetLastError());
		cuCHECK(cudaDeviceSynchronize());
	}
}