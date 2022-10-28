#pragma once

#include "HilbertSpace.hpp"

#ifndef __NVCC__
	#define __host__
	#define __device__
#endif

template<class Derived>
struct ManyBodySpaceTraits;

template<class Derived>
class ManyBodySpaceBase : public HilbertSpace<Derived> {
	private:
		using LocalSpace = typename ManyBodySpaceTraits<Derived>::LocalSpace;

		int        m_sysSize = 0;
		LocalSpace m_locSpace;

	public:
		/**
		 * @brief Constructor1
		 *
		 * @param systemSize
		 * @param locSpace
		 * @return __host__
		 */
		__host__ __device__ ManyBodySpaceBase(int m_sysSize, LocalSpace const& locSpace)
		    : m_sysSize{m_sysSize}, m_locSpace{locSpace} {}

		__host__ __device__ int               sysSize() const { return m_sysSize; }
		__host__ __device__ LocalSpace const& locSpace() const { return m_locSpace; }
		__host__ __device__ int               dimLoc() const { return m_locSpace.dim(); }
};

class ManyBodySpinSpace;
template<>
struct ManyBodySpaceTraits<ManyBodySpinSpace> {
		using LocalSpace = HilbertSpace<int>;
};
class ManyBodySpinSpace : public ManyBodySpaceBase<ManyBodySpinSpace> {
	private:
		using Base       = ManyBodySpaceBase<ManyBodySpinSpace>;
		using LocalSpace = typename ManyBodySpaceTraits<ManyBodySpinSpace>::LocalSpace;

	public:
		/**
		 * @brief Constructor1
		 *
		 * @param sysSize
		 * @param locSpace
		 */
		__host__ __device__ ManyBodySpinSpace(int sysSize, LocalSpace const& locSpace)
		    : Base(sysSize, locSpace) {}

		/**
		 * @brief Default constructor
		 *
		 * @param sysSize
		 * @param locSpace
		 */
		__host__ __device__ ManyBodySpinSpace(int sysSize = 0, int dimLoc = 0)
		    : Base(sysSize, LocalSpace(dimLoc)) {}

	private:
		friend HilbertSpace<ManyBodySpinSpace>;
		__host__ __device__ int dim_impl() const {
			int res = 1;
			for(int l = 0; l != this->sysSize(); ++l) { res *= this->dimLoc(); }
			return res;
		}
};