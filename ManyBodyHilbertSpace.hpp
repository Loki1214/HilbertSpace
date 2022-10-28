#pragma once

#include "HilbertSpace.hpp"
#include <Eigen/Dense>

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

		size_t     m_sysSize = 0;
		LocalSpace m_locSpace;

	public:
		/**
		 * @brief Constructor1
		 *
		 * @param systemSize
		 * @param locSpace
		 * @return __host__
		 */
		__host__ __device__ ManyBodySpaceBase(size_t sysSize, LocalSpace const& locSpace)
		    : m_sysSize{sysSize}, m_locSpace{locSpace} {}

		__host__ __device__ ManyBodySpaceBase()                                          = default;
		__host__ __device__ ManyBodySpaceBase(ManyBodySpaceBase const& other)            = default;
		__host__ __device__ ManyBodySpaceBase& operator=(ManyBodySpaceBase const& other) = delete;
		__host__ __device__ ManyBodySpaceBase(ManyBodySpaceBase&& other)                 = default;
		__host__ __device__ ManyBodySpaceBase& operator=(ManyBodySpaceBase&& other)      = delete;
		__host__                               __device__ ~ManyBodySpaceBase()           = default;

		__host__ __device__ size_t            sysSize() const { return m_sysSize; }
		__host__ __device__ LocalSpace const& locSpace() const { return m_locSpace; }
		__host__ __device__ size_t            dimLoc() const { return m_locSpace.dim(); }

		// Statically polymorphic functions
		__host__ __device__ int locState(int stateNum, int pos) const {
			return static_cast<Derived const*>(this)->locState_impl(stateNum, pos);
		}
		__host__ __device__ Eigen::RowVectorXi ordinalToConfig(size_t stateNum) const {
			return static_cast<Derived const*>(this)->ordinalToConfig_impl(stateNum);
		}
		template<class Array>
		__host__ __device__ size_t configToOrdinal(Array const& config) const {
			return static_cast<Derived const*>(this)->configToOrdinal_impl(config);
		}
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
		__host__ __device__ ManyBodySpinSpace(size_t sysSize, LocalSpace const& locSpace)
		    : Base(sysSize, locSpace) {}

		/**
		 * @brief Default constructor
		 *
		 * @param sysSize
		 * @param locSpace
		 */
		__host__ __device__ ManyBodySpinSpace(size_t sysSize = 0, size_t dimLoc = 0)
		    : Base(sysSize, LocalSpace(dimLoc)) {}

	private:
		friend HilbertSpace<ManyBodySpinSpace>;
		__host__ __device__ size_t dim_impl() const {
			if(this->sysSize() == 0) return 0;
			size_t res = 1;
			for(size_t l = 0; l != this->sysSize(); ++l) { res *= this->dimLoc(); }
			return res;
		}

		friend ManyBodySpaceBase<ManyBodySpinSpace>;
		__host__ __device__ int locState_impl(int stateNum, int pos) const {
			assert(0 <= pos && pos < static_cast<int>(this->sysSize()));
			for(auto l = 0; l != pos; ++l) stateNum /= this->dimLoc();
			return stateNum % this->dimLoc();
		}
		__host__ __device__ Eigen::RowVectorXi ordinalToConfig_impl(int stateNum) const {
			Eigen::RowVectorXi res(this->sysSize());
			for(size_t l = 0; l != this->sysSize(); ++l, stateNum /= this->dimLoc()) {
				res(l) = stateNum % this->dimLoc();
			}
			return res;
		}
		template<class Array>
		__host__ __device__ size_t configToOrdinal_impl(Array const& config) const {
			assert(config.size() >= static_cast<int>(this->sysSize()));
			size_t res  = 0;
			size_t base = 1;
			for(size_t l = 0; l != this->sysSize(); ++l, base *= this->dimLoc()) {
				res += config(l) * base;
			}
			return res;
		}
};