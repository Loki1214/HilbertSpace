#pragma once

#include "typedefs.hpp"
#include "ManyBodySpaceBase.hpp"
#include <cassert>

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
#endif  // ifndef cuCHECK

template<class Derived>
__global__ void check_bijectiveness_kernel(ManyBodySpaceBase<Derived> const* mbSpacePtr);
template<class Derived>
__global__ void check_transEqClass_kernel(ManyBodySpaceBase<Derived> const* mbSpacePtr,
                                          int*                              appeared);
template<class Derived>
__global__ void check_reverseOp_kernel(ManyBodySpaceBase<Derived> const* mbSpacePtr);

template<class Derived, class LocalSpace>
__device__ void test_ManyBodySpaceBase(ManyBodySpaceBase<Derived> const* mbSpacePtr, Size sysSize,
                                       LocalSpace const* locSpacePtr) {
	auto const& mbSpace  = *mbSpacePtr;
	auto const& locSpace = *locSpacePtr;
	printf("\tmbSpace.dim() = %lu\n", static_cast<unsigned long>(mbSpace.dim()));
	assert(sysSize != 0 || mbSpace.dim() == 0);
	assert(mbSpace.sysSize() == sysSize);
	assert(mbSpace.dimLoc() == locSpace.dim());
	assert(mbSpace.locSpace() == locSpace);
	if(mbSpace.dim() == 0) return;

	constexpr int threads = 512;
	int           grids;
	auto          upperQuotient = [&](int x, int y) { return (x % y == 0 ? x / y : x / y + 1); };

	// test locState
	// test ordinal_to_config
	// test config_to_ordinal
	grids = upperQuotient(mbSpace.dim(), threads);
	check_bijectiveness_kernel<<<grids, threads>>>(mbSpacePtr);
	cuCHECK(cudaGetLastError());
	cuCHECK(cudaDeviceSynchronize());

	// test for translation operations
	mbSpace.compute_transEqClass();
	printf("\tmbSpace.transEqDim() = %lu\n", static_cast<unsigned long>(mbSpace.transEqDim()));
	Eigen::ArrayXi appeared = Eigen::ArrayXi::Zero(mbSpace.dim());

	grids = upperQuotient(mbSpace.transEqDim(), threads);
	check_transEqClass_kernel<<<grids, threads>>>(mbSpacePtr, appeared.data());
	cuCHECK(cudaGetLastError());
	cuCHECK(cudaDeviceSynchronize());
	for(Size stateNum = 0; stateNum != mbSpace.dim(); ++stateNum) assert(appeared[stateNum] == 1);

	// test for state_to_transEqClass
	// test for state_to_transShift
	// #pragma omp parallel for
	// 	for(auto sample = 0; sample < index.size(); ++sample) {
	// 		Size       stateNum   = index(sample);
	// 		auto const eqClass    = mbSpace.state_to_transEqClass(stateNum);
	// 		auto const eqClassRep = mbSpace.transEqClassRep(eqClass);
	// 		auto const trans      = mbSpace.state_to_transShift(stateNum);
	// 		assert(stateNum == mbSpace.translate(eqClassRep, trans));
	// 	}

	// test for reverse()
	grids = upperQuotient(mbSpace.dim(), threads);
	check_reverseOp_kernel<<<grids, threads>>>(mbSpacePtr);
	cuCHECK(cudaGetLastError());
	cuCHECK(cudaDeviceSynchronize());
}

template<class Derived>
__global__ void check_bijectiveness_kernel(ManyBodySpaceBase<Derived> const* mbSpacePtr) {
	auto const& mbSpace  = *mbSpacePtr;
	Size const  stateNum = blockIdx.x * blockDim.x + threadIdx.x;
	if(stateNum >= mbSpace.dim()) return;

	auto config = mbSpace.ordinal_to_config(stateNum);
	assert(stateNum == mbSpace.config_to_ordinal(config));
	for(Size pos = 0; pos != mbSpace.sysSize(); ++pos) {
		assert(config(pos) == mbSpace.locState(stateNum, pos));
	}
}

template<class Derived>
__global__ void check_transEqClass_kernel(ManyBodySpaceBase<Derived> const* mbSpacePtr,
                                          int*                              appeared) {
	auto const& mbSpace    = *mbSpacePtr;
	Size const  eqClassNum = blockIdx.x * blockDim.x + threadIdx.x;
	if(eqClassNum >= mbSpace.transEqDim()) return;

	auto const stateNum = mbSpace.transEqClassRep(eqClassNum);
	atomicAdd(&appeared[stateNum], 1);
	for(auto trans = 1; trans != mbSpace.transPeriod(eqClassNum); ++trans) {
		auto translated = mbSpace.translate(stateNum, trans);
		atomicAdd(&appeared[translated], 1);
	}
}

template<class Derived>
__global__ void check_reverseOp_kernel(ManyBodySpaceBase<Derived> const* mbSpacePtr) {
	auto const& mbSpace  = *mbSpacePtr;
	Size const  stateNum = blockIdx.x * blockDim.x + threadIdx.x;
	if(stateNum >= mbSpace.dim()) return;

	auto config   = mbSpace.ordinal_to_config(stateNum);
	auto reversed = mbSpace.config_to_ordinal(config.reverse());
	assert(reversed == mbSpace.reverse(stateNum));
}