#pragma once

#include "typedefs.hpp"
#include "ManyBodySpaceBase.hpp"

class ManyBodySpinSpace;
template<>
struct ManyBodySpaceTraits<ManyBodySpinSpace> {
		using LocalSpace = HilbertSpace<int>;
};
class ManyBodySpinSpace : public ManyBodySpaceBase<ManyBodySpinSpace> {
	private:
		using Base = ManyBodySpaceBase<ManyBodySpinSpace>;

	public:
		using LocalSpace = typename Base::LocalSpace;

	public:
		/**
		 * @brief Constructor1
		 *
		 * @param sysSize
		 * @param locSpace
		 */
		__host__ __device__ ManyBodySpinSpace(Size sysSize, LocalSpace const& locSpace)
		    : Base(sysSize, locSpace) {}

		/**
		 * @brief Default constructor
		 *
		 * @param sysSize
		 * @param locSpace
		 */
		__host__ __device__ ManyBodySpinSpace(Size sysSize = 0, Size dimLoc = 0)
		    : ManyBodySpinSpace(sysSize, LocalSpace(dimLoc)) {}

	private:
		/*! @name Implementation for methods of ancestor class HilbertSpace */
		/* @{ */
		friend HilbertSpace<ManyBodySpinSpace>;
		__host__ __device__ Size dim_impl() const {
			if(this->sysSize() == 0) return 0;
			Size res = 1;
			for(Size l = 0; l != this->sysSize(); ++l) { res *= this->dimLoc(); }
			return res;
		}
		/* @} */

		/*! @name Implementation for methods of parent class ManyBodySpaceBase */
		/* @{ */
		friend ManyBodySpaceBase<ManyBodySpinSpace>;
		__host__ __device__ Size locState_impl(Size stateNum, int pos) const {
			assert(stateNum < this->dim());
			assert(pos < this->sysSize());
			for(auto l = 0; l != pos; ++l) stateNum /= this->dimLoc();
			return stateNum % this->dimLoc();
		}

		__host__ __device__ Eigen::RowVectorX<Size> ordinal_to_config_impl(Size stateNum) const {
			assert(stateNum < this->dim());
			Eigen::RowVectorX<Size> res(this->sysSize());
			for(Size l = 0; l != this->sysSize(); ++l, stateNum /= this->dimLoc()) {
				res(l) = stateNum % this->dimLoc();
			}
			return res;
		}

		template<class Array>
		__host__ __device__ Size config_to_ordinal_impl(Array const& config) const {
			assert(static_cast<Size>(config.size()) >= this->sysSize());
			Size res  = 0;
			Size base = 1;
			for(Size l = 0; l != this->sysSize(); ++l, base *= this->dimLoc()) {
				res += config(l) * base;
			}
			return res;
		}

		template<typename... Args>
		__host__ __device__ Size translate_impl(Size stateNum, int trans, Args...) const {
			assert(stateNum < this->dim());
			assert(0 <= trans && trans < this->sysSize());
			Size base = 1;
			for(auto l = 0; l != trans; ++l) base *= this->dimLoc();
			Size const baseCompl = this->dim() / base;
			return stateNum / baseCompl + (stateNum % baseCompl) * base;
		}

		__host__ __device__ Size reverse_impl(Size stateNum) const {
			assert(stateNum < this->dim());
			Size res = 0, base = 1;
			for(Size l = 0; l != this->sysSize(); ++l, base *= this->dimLoc()) {
				res += (this->dim() / base / this->dimLoc()) * ((stateNum / base) % this->dimLoc());
			}
			return res;
		}
		/* @} */
};