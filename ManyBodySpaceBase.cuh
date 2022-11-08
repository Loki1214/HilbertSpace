#pragma once

#include "typedefs.hpp"
#include "ManyBodySpaceBase.hpp"

template<class Derived>
__global__ void compute_transEqClass_kernel(ManyBodySpaceBase<Derived> const* mbSpacePtr,
                                            Size* transEqClassRep, int* transPeriod,
                                            bool* calculated) {
	auto const& mbSpace  = *mbSpacePtr;
	long const  stateNum = blockIdx.x * blockDim.x + threadIdx.x;
	if(stateNum >= mbSpace.dim()) return;
	if(calculated[stateNum]) return;
	calculated[stateNum] = true;

	bool                   duplicationFlag = false;
	Size                   trans, eqClassRep = stateNum;
	extern __shared__ Size sharedMem[];
	Size*                  translated = sharedMem + threadIdx.x * mbSpace.sysSize();

	translated[0] = stateNum;
	for(trans = 1; trans < mbSpace.sysSize(); ++trans) {
		auto const transed = mbSpace.translate(stateNum, trans);
		translated[trans]  = transed;
		if(transed == stateNum) break;
		eqClassRep = (transed < eqClassRep ? transed : eqClassRep);
		if(transed == eqClassRep && calculated[transed]) {
			duplicationFlag = true;
			break;
		}
		calculated[transed] = true;
	}
	if(duplicationFlag) return;

	auto const period = trans;
	for(trans = 0; trans != period; ++trans) {
		auto const idx       = translated[trans];
		transEqClassRep[idx] = eqClassRep;
		transPeriod[idx]     = period;
	}
}

__global__ void make_paired_vector_kernel(Eigen::ArrayX<std::pair<Size, int>>* outVecPtr,
                                          Size* transEqClassRep, int* transPeriod) {
	auto&      outVec     = *outVecPtr;
	Size const eqClassNum = blockIdx.x * blockDim.x + threadIdx.x;
	if(eqClassNum >= outVec.size()) return;
	outVec(eqClassNum).first  = transEqClassRep[eqClassNum];
	outVec(eqClassNum).second = transPeriod[eqClassNum];
}