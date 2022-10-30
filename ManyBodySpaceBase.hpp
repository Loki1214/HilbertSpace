#pragma once

#include "HilbertSpace.hpp"
#include <Eigen/Dense>

#ifndef __NVCC__
	#define __host__
	#define __device__
#endif

template<class Derived>
struct ManyBodySpaceTraits;
// OpSpaceTraits should define the following properties:
// - LocalSpace

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
		__host__ __device__ size_t locState(size_t stateNum, int pos) const {
			return static_cast<Derived const*>(this)->locState_impl(stateNum, pos);
		}
		__host__ __device__ Eigen::RowVectorX<size_t> ordinal_to_config(size_t stateNum) const {
			return static_cast<Derived const*>(this)->ordinal_to_config_impl(stateNum);
		}
		template<class Array>
		__host__ __device__ size_t config_to_ordinal(Array const& config) const {
			return static_cast<Derived const*>(this)->config_to_ordinal_impl(config);
		}

		/*! @name Translation operation */
		/* @{ */

	private:
		mutable Eigen::ArrayX<size_t> m_transEqClassRep;
		mutable Eigen::ArrayX<size_t> m_transPeriod;
		mutable Eigen::ArrayX<size_t> m_stateToTransEqClass;
		mutable Eigen::ArrayX<size_t> m_stateToTransShift;

	public:
		__host__ void compute_transEqClass() const;

		__host__ __device__ size_t transEqDim() const { return m_transEqClassRep.size(); }

		__host__ __device__ Eigen::ArrayX<size_t> const& transEqClassRep() const {
			return m_transEqClassRep;
		}
		__host__ __device__ size_t transEqClassRep(size_t eqClassNum) const {
			return m_transEqClassRep(eqClassNum);
		}

		__host__ __device__ Eigen::ArrayX<size_t> const& transPeriod() const {
			return m_transPeriod;
		}
		__host__ __device__ size_t transPeriod(size_t eqClassNum) const {
			return m_transPeriod(eqClassNum);
		}

		__host__ __device__ size_t state_to_transEqClass(size_t stateNum) const {
			return m_stateToTransEqClass(stateNum);
		}
		__host__ __device__ size_t state_to_transPeriod(size_t stateNum) const {
			auto eqClass = this->state_to_transEqClass(stateNum);
			return this->transPeriod(eqClass);
		}
		__host__ __device__ size_t state_to_transShift(size_t stateNum) const {
			return m_stateToTransShift(stateNum);
		}

		// Statically polymorphic functions
		__host__ __device__ size_t translate(size_t stateNum, int trans) const {
			return static_cast<Derived const*>(this)->translate_impl(stateNum, trans);
		}
		template<class Array>
		__host__ __device__ size_t translate(size_t stateNum, int trans, Array& work) const {
			return static_cast<Derived const*>(this)->translate_impl(stateNum, trans, work);
		}
		/* @} */

		/*! @name Parity operation */
		/* @{ */

	public:
		// Statically polymorphic functions
		__host__ __device__ size_t reverse(size_t stateNum) const {
			return static_cast<Derived const*>(this)->reverse_impl(stateNum);
		}
		/* @} */
};

template<class Derived>
__host__ void ManyBodySpaceBase<Derived>::compute_transEqClass() const {
	if(m_transEqClassRep.size() >= 1) return;
	if(this->dim() <= 0) return;

	Eigen::ArrayX<bool> calculated = Eigen::ArrayX<bool>::Zero(this->dim());
	size_t              eqClassNum = 0;
	size_t              period, translated;
	for(size_t stateNum = 0; stateNum < this->dim(); ++stateNum) {
		if(calculated(stateNum)) continue;
		calculated(stateNum) = true;
		for(period = 1; period < this->sysSize(); ++period) {
			translated = this->translate(stateNum, period);
			if(translated == stateNum) break;
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
			if(translated == stateNum) break;
			calculated(translated)            = true;
			m_stateToTransEqClass(translated) = eqClassNum;
			m_stateToTransShift(translated)   = period;
		}
		m_transEqClassRep(eqClassNum) = stateNum;
		m_transPeriod(eqClassNum)     = period;
		eqClassNum += 1;
	}
	assert(eqClassNum == static_cast<size_t>(m_transEqClassRep.size()));
}