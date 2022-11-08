// #include <catch2/catch_test_macros.hpp>
#include "ManyBodySpaceBase_test.cuh"
#include "ManyBodyHilbertSpace/ManyBodySpinSpace.hpp"
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

__global__ void test_ManyBodySpinSpace_kernel(Size dimLoc) {
	{
		// Default constructor
		ManyBodySpinSpace* mbSpacePtr  = new ManyBodySpinSpace;
		HilbertSpace<int>* locSpacePtr = new HilbertSpace<int>();
		test_ManyBodySpaceBase(mbSpacePtr, 0, locSpacePtr);
		delete mbSpacePtr;
		delete locSpacePtr;
	}
	{
		// test Constructor1
		HilbertSpace<int>* locSpacePtr = new HilbertSpace<int>(dimLoc);
		ManyBodySpinSpace* mbSpacePtr  = new ManyBodySpinSpace(0, *locSpacePtr);
		test_ManyBodySpaceBase(mbSpacePtr, 0, locSpacePtr);
		delete mbSpacePtr;
		for(Size sysSize = 1; sysSize <= 16; ++sysSize) {
			ManyBodySpinSpace* mbSpacePtr = new ManyBodySpinSpace(sysSize, *locSpacePtr);
			test_ManyBodySpaceBase(mbSpacePtr, sysSize, locSpacePtr);
			delete mbSpacePtr;
		}
	}
}

// TEST_CASE("ManyBodySpinSpace_onGPU", "test") {
int main() {
	Size dimLoc = 2;
	test_ManyBodySpinSpace_kernel<<<1, 1>>>(dimLoc);
	cuCHECK(cudaGetLastError());
	cuCHECK(cudaDeviceSynchronize());
}