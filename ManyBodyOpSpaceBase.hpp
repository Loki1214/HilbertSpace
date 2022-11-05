#pragma once

#include "ManyBodySpaceBase.hpp"
#include "OpSpaceBase.hpp"

#ifndef __NVCC__
	#define __host__
	#define __device__
#endif

template<class Derived>
class ManyBodyOpSpaceBase : public OpSpaceBase<Derived>,
                            public ManyBodySpaceBase<Derived> {
	public:
		using BaseSpace  = typename OpSpaceBase<Derived>::BaseSpace;
		using Scalar     = typename OpSpaceBase<Derived>::Scalar;
		using RealScalar = typename OpSpaceBase<Derived>::RealScalar;
		using LocalSpace = typename ManyBodySpaceBase<Derived>::LocalSpace;

	public:
		using ManyBodySpaceBase<Derived>::dim;

		/**
		 * @brief Custom constructor 1
		 *
		 * @param baseSpace
		 * @param sysSize
		 * @param locSpace
		 */
		__host__ __device__ ManyBodyOpSpaceBase(BaseSpace const& baseSpace, size_t sysSize,
		                                        LocalSpace const& locSpace)
		    : OpSpaceBase<Derived>(baseSpace), ManyBodySpaceBase<Derived>(sysSize, locSpace) {}
		__host__ __device__ ManyBodyOpSpaceBase(BaseSpace&& baseSpace, size_t sysSize,
		                                        LocalSpace&& locSpace)
		    : OpSpaceBase<Derived>(std::move(baseSpace)),
		      ManyBodySpaceBase<Derived>(sysSize, std::move(locSpace)) {}

		__host__ __device__ ManyBodyOpSpaceBase()                                      = default;
		__host__ __device__ ManyBodyOpSpaceBase(ManyBodyOpSpaceBase const&)            = default;
		__host__ __device__ ManyBodyOpSpaceBase& operator=(ManyBodyOpSpaceBase const&) = default;
		__host__ __device__ ManyBodyOpSpaceBase(ManyBodyOpSpaceBase&&)                 = default;
		__host__ __device__ ManyBodyOpSpaceBase& operator=(ManyBodyOpSpaceBase&&)      = default;
		__host__                                 __device__ ~ManyBodyOpSpaceBase()     = default;
};