#pragma once
#include "../ManyBodySpaceBase.hpp"

class ManyBodyBosonSpace;
template<>
struct ManyBodySpaceTraits<ManyBodyBosonSpace> {
		using LocalSpace = HilbertSpace<int>;
};
class ManyBodyBosonSpace : public ManyBodySpaceBase<ManyBodyBosonSpace> {
	private:
		using Base       = ManyBodySpaceBase<ManyBodyBosonSpace>;
		using LocalSpace = typename ManyBodySpaceTraits<ManyBodyBosonSpace>::LocalSpace;
		IntegerComposition m_iComp;

	public:
		/**
		 * @brief Default constructor
		 *
		 * @param sysSize
		 * @param nBosons
		 */
		__host__ __device__ ManyBodyBosonSpace(size_t sysSize = 0, size_t nBosons = 0)
		    : Base(sysSize, LocalSpace(nBosons + 1)) {}

	// private:
	// 	/*! @name Implementation for functions of ancestor class HilbertSpace */
	// 	/* @{ */
	// 	friend HilbertSpace<ManyBodyBosonSpace>;
	// 	__host__ __device__ size_t dim_impl() const {
	// 		if(this->sysSize() == 0) return 0;
	// 		size_t res = 1;
	// 		for(size_t l = 0; l != this->sysSize(); ++l) { res *= this->dimLoc(); }
	// 		return res;
	// 	}
	// 	/* @} */

	// 	/*! @name Implementation for functions of parent class ManyBodySpaceBase */
	// 	/* @{ */
	// 	friend ManyBodySpaceBase<ManyBodyBosonSpace>;
	// 	__host__ __device__ int locState_impl(int stateNum, int pos) const {
	// 		assert(0 <= pos && pos < static_cast<int>(this->sysSize()));
	// 		for(auto l = 0; l != pos; ++l) stateNum /= this->dimLoc();
	// 		return stateNum % this->dimLoc();
	// 	}

	// 	__host__ __device__ Eigen::RowVectorXi ordinal_to_config_impl(int stateNum) const {
	// 		Eigen::RowVectorXi res(this->sysSize());
	// 		for(size_t l = 0; l != this->sysSize(); ++l, stateNum /= this->dimLoc()) {
	// 			res(l) = stateNum % this->dimLoc();
	// 		}
	// 		return res;
	// 	}

	// 	template<class Array>
	// 	__host__ __device__ size_t config_to_ordinal_impl(Array const& config) const {
	// 		assert(config.size() >= static_cast<int>(this->sysSize()));
	// 		size_t res  = 0;
	// 		size_t base = 1;
	// 		for(size_t l = 0; l != this->sysSize(); ++l, base *= this->dimLoc()) {
	// 			res += config(l) * base;
	// 		}
	// 		return res;
	// 	}

	// 	template<typename... Args>
	// 	__host__ __device__ int translate_impl(int stateNum, int trans, Args...) const {
	// 		assert(0 <= stateNum && stateNum < static_cast<int>(this->dim()));
	// 		assert(0 <= trans && trans < static_cast<int>(this->sysSize()));
	// 		size_t base = 1;
	// 		for(auto l = 0; l != trans; ++l) base *= this->dimLoc();
	// 		size_t const baseCompl = this->dim() / base;
	// 		return stateNum / baseCompl + (stateNum % baseCompl) * base;
	// 	}

	// 	__host__ __device__ int reverse_impl(int stateNum) const {
	// 		assert(0 <= stateNum && stateNum < static_cast<int>(this->dim()));
	// 		int    res  = 0;
	// 		size_t base = 1;
	// 		for(size_t l = 0; l != this->sysSize(); ++l, base *= this->dimLoc()) {
	// 			res += (this->dim() / base / this->dimLoc()) * ((stateNum / base) % this->dimLoc());
	// 		}
	// 		return res;
	// 	}
	// 	/* @} */
};