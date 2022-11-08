// #include <catch2/catch_test_macros.hpp>
#include "ManyBodySpaceBase_test.cuh"
#include "ManyBodyHilbertSpace/ManyBodyBosonSpace.hpp"
#include <iostream>

#ifndef cuCHECK
	#define cuCHECK(call)                                                          \
		{                                                                          \
			const cudaError_t error = call;                                        \
			if(error != cudaSuccess) {                                             \
				printf("cuCHECK Error: %s:%d,  ", __FILE__, __LINE__);             \
				printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
				assert(error == cudaSuccess);                                      \
			}                                                                      \
		};
#endif

__global__ void test_ManyBodyBosonSpace_kernel(Size dimLoc, Size LMax, Size NMax) {
	{
		// Default constructor
		ManyBodyBosonSpace* mbSpacePtr  = new ManyBodyBosonSpace;
		HilbertSpace<int>*  locSpacePtr = new HilbertSpace<int>();
		test_ManyBodySpaceBase(mbSpacePtr, 0, locSpacePtr);
		delete mbSpacePtr;
		delete locSpacePtr;
	}
	{
		// test Constructor1
		// HilbertSpace<int>*  locSpacePtr = new HilbertSpace<int>(dimLoc);
		// ManyBodyBosonSpace* mbSpacePtr  = new ManyBodyBosonSpace(0, 0, *locSpacePtr);
		// test_ManyBodySpaceBase(mbSpacePtr, 0, locSpacePtr);
		// delete mbSpacePtr;
		// delete locSpacePtr;

		for(Size L = 1; L <= LMax; ++L)
			for(Size N = 1; N <= min(NMax,L); ++N) {
				printf("\tL = %ld, N = %ld\n", L, N);
				auto* locSpacePtr = new HilbertSpace<int>(N + 1);
				auto* mbSpacePtr  = new ManyBodyBosonSpace(L, N, *locSpacePtr);
				test_ManyBodySpaceBase(mbSpacePtr, L, locSpacePtr);
				delete mbSpacePtr;
				printf("\n");
			}
	}
}

// TEST_CASE("ManyBodyBosonSpace_onGPU", "test") {
int main() {
	size_t pValue;
	cuCHECK(cudaDeviceGetLimit(&pValue, cudaLimitMallocHeapSize));
	std::cout << "cudaLimitMallocHeapSize = " << pValue << std::endl;
	pValue *= 16;
	cuCHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, pValue));
	cuCHECK(cudaDeviceGetLimit(&pValue, cudaLimitMallocHeapSize));
	std::cout << "cudaLimitMallocHeapSize = " << pValue << std::endl;

	constexpr Size dimLoc = 2;
	constexpr Size LMax   = 16;
	constexpr Size NMax   = 6;
	test_ManyBodyBosonSpace_kernel<<<1, 1>>>(dimLoc, LMax, NMax);
	cuCHECK(cudaGetLastError());
	cuCHECK(cudaDeviceSynchronize());
}