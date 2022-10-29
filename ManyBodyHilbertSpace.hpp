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
		 * @brief Custom constructor
		 *
		 * @param systemSize
		 * @param locSpace
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
		__host__ __device__ Eigen::RowVectorXi ordinal_to_config(size_t stateNum) const {
			return static_cast<Derived const*>(this)->ordinal_to_config_impl(stateNum);
		}
		template<class Array>
		__host__ __device__ size_t config_to_ordinal(Array const& config) const {
			return static_cast<Derived const*>(this)->config_to_ordinal_impl(config);
		}

		/*! @name Translation operation */
		/* @{ */

	private:
		mutable Eigen::ArrayXi m_transEqClassRep;
		mutable Eigen::ArrayXi m_transPeriod;
		mutable Eigen::ArrayXi m_stateToTransEqClass;
		mutable Eigen::ArrayXi m_stateToTransShift;

	public:
		__host__ void compute_transEqClass() const;

		__host__ __device__ size_t transEqDim() const { return m_transEqClassRep.size(); }

		__host__ __device__ Eigen::ArrayXi const& transEqClassRep() const {
			return m_transEqClassRep;
		}
		__host__ __device__ int transEqClassRep(int eqClassNum) const {
			return m_transEqClassRep(eqClassNum);
		}

		__host__ __device__ Eigen::ArrayXi const& transPeriod() const { return m_transPeriod; }
		__host__ __device__ int                   transPeriod(int eqClassNum) const {
			                  return m_transPeriod(eqClassNum);
		}

		__host__ __device__ int state_to_transEqClass(int stateNum) const {
			return m_stateToTransEqClass(stateNum);
		}
		__host__ __device__ int state_to_transPeriod(int stateNum) const {
			auto eqClass = this->state_to_transEqClass(stateNum);
			return this->transPeriod(eqClass);
		}
		__host__ __device__ int state_to_transShift(int stateNum) const {
			return m_stateToTransShift(stateNum);
		}

		// Statically polymorphic functions
		__host__ __device__ int translate(int stateNum, int trans) const {
			return static_cast<Derived const*>(this)->translate_impl(stateNum, trans);
		}
		template<class Array>
		__host__ __device__ int translate(int stateNum, int trans, Array& work) const {
			return static_cast<Derived const*>(this)->translate_impl(stateNum, trans, work);
		}
		/* @} */

		/*! @name Parity operation */
		/* @{ */

	public:
		// Statically polymorphic functions
		__host__ __device__ int reverse(int stateNum) const {
			return static_cast<Derived const*>(this)->reverse_impl(stateNum);
		}
		/* @} */
};

template<class Derived>
__host__ void ManyBodySpaceBase<Derived>::compute_transEqClass() const {
	if(m_transEqClassRep.size() >= 1) return;
	if(this->dim() <= 0) return;

	Eigen::ArrayX<bool> calculated = Eigen::ArrayX<bool>::Zero(this->dim());
	int                 eqClassNum = 0;
	size_t              period;
	int                 translated;
	for(size_t stateNum = 0; stateNum < this->dim(); ++stateNum) {
		if(calculated(stateNum)) continue;
		calculated(stateNum) = true;
		for(period = 1; period < this->sysSize(); ++period) {
			translated = this->translate(stateNum, period);
			if(translated == static_cast<int>(stateNum)) break;
			calculated(translated) = true;
		}
		eqClassNum += 1;
	}

	m_transEqClassRep.resize(eqClassNum);
	m_transPeriod.resize(eqClassNum);
	m_stateToTransEqClass.resize(this->dim());
	m_stateToTransShift.resize(this->dim());
	calculated = Eigen::ArrayX<bool>::Zero(this->dim());
	eqClassNum = 0;
	for(size_t stateNum = 0; stateNum < this->dim(); ++stateNum) {
		if(calculated(stateNum)) continue;
		calculated(stateNum)            = true;
		m_stateToTransEqClass(stateNum) = eqClassNum;
		m_stateToTransShift(stateNum)   = 0;

		for(period = 1; period < this->sysSize(); ++period) {
			translated = this->translate(stateNum, period);
			if(translated == static_cast<int>(stateNum)) break;
			calculated(translated)            = true;
			m_stateToTransEqClass(translated) = eqClassNum;
			m_stateToTransShift(translated)   = period;
		}
		m_transEqClassRep(eqClassNum) = stateNum;
		m_transPeriod(eqClassNum)     = period;
		eqClassNum += 1;
	}
	assert(eqClassNum == m_transEqClassRep.size());
}

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
		/*! @name Implementation for functions of ancestor class HilbertSpace */
		/* @{ */
		friend HilbertSpace<ManyBodySpinSpace>;
		__host__ __device__ size_t dim_impl() const {
			if(this->sysSize() == 0) return 0;
			size_t res = 1;
			for(size_t l = 0; l != this->sysSize(); ++l) { res *= this->dimLoc(); }
			return res;
		}
		/* @} */

		/*! @name Implementation for functions of parent class ManyBodySpaceBase */
		/* @{ */
		friend ManyBodySpaceBase<ManyBodySpinSpace>;
		__host__ __device__ int locState_impl(int stateNum, int pos) const {
			assert(0 <= pos && pos < static_cast<int>(this->sysSize()));
			for(auto l = 0; l != pos; ++l) stateNum /= this->dimLoc();
			return stateNum % this->dimLoc();
		}

		__host__ __device__ Eigen::RowVectorXi ordinal_to_config_impl(int stateNum) const {
			Eigen::RowVectorXi res(this->sysSize());
			for(size_t l = 0; l != this->sysSize(); ++l, stateNum /= this->dimLoc()) {
				res(l) = stateNum % this->dimLoc();
			}
			return res;
		}

		template<class Array>
		__host__ __device__ size_t config_to_ordinal_impl(Array const& config) const {
			assert(config.size() >= static_cast<int>(this->sysSize()));
			size_t res  = 0;
			size_t base = 1;
			for(size_t l = 0; l != this->sysSize(); ++l, base *= this->dimLoc()) {
				res += config(l) * base;
			}
			return res;
		}

		template<typename... Args>
		__host__ __device__ int translate_impl(int stateNum, int trans, Args...) const {
			assert(0 <= stateNum && stateNum < static_cast<int>(this->dim()));
			assert(0 <= trans && trans < static_cast<int>(this->sysSize()));
			size_t base = 1;
			for(auto l = 0; l != trans; ++l) base *= this->dimLoc();
			size_t const baseCompl = this->dim() / base;
			return stateNum / baseCompl + (stateNum % baseCompl) * base;
		}

		__host__ __device__ int reverse_impl(int stateNum) const {
			assert(0 <= stateNum && stateNum < static_cast<int>(this->dim()));
			int    res  = 0;
			size_t base = 1;
			for(size_t l = 0; l != this->sysSize(); ++l, base *= this->dimLoc()) {
				res += (this->dim() / base / this->dimLoc()) * ((stateNum / base) % this->dimLoc());
			}
			return res;
		}
		/* @} */
};