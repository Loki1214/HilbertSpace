#pragma once
#include "../ManyBodySpaceBase.hpp"

class ManyBodySpinSpace;
template<>
struct ManyBodySpaceTraits<ManyBodySpinSpace> {
		using LocalSpace = HilbertSpace<int>;
};
class ManyBodySpinSpace : public ManyBodySpaceBase<ManyBodySpinSpace> {
	private:
		using Base       = ManyBodySpaceBase<ManyBodySpinSpace>;

	public:
		using LocalSpace = typename Base::LocalSpace;

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
		    : ManyBodySpinSpace(sysSize, LocalSpace(dimLoc)) {}

	private:
		/*! @name Implementation for methods of ancestor class HilbertSpace */
		/* @{ */
		friend HilbertSpace<ManyBodySpinSpace>;
		__host__ __device__ size_t dim_impl() const {
			if(this->sysSize() == 0) return 0;
			size_t res = 1;
			for(size_t l = 0; l != this->sysSize(); ++l) { res *= this->dimLoc(); }
			return res;
		}
		/* @} */

		/*! @name Implementation for methods of parent class ManyBodySpaceBase */
		/* @{ */
		friend ManyBodySpaceBase<ManyBodySpinSpace>;
		__host__ __device__ size_t locState_impl(size_t stateNum, int pos) const {
			assert(stateNum < this->dim());
			assert(static_cast<size_t>(pos) < this->sysSize());
			for(auto l = 0; l != pos; ++l) stateNum /= this->dimLoc();
			return stateNum % this->dimLoc();
		}

		__host__ __device__ Eigen::RowVectorX<size_t> ordinal_to_config_impl(
		    size_t stateNum) const {
			assert(stateNum < this->dim());
			Eigen::RowVectorX<size_t> res(this->sysSize());
			for(size_t l = 0; l != this->sysSize(); ++l, stateNum /= this->dimLoc()) {
				res(l) = stateNum % this->dimLoc();
			}
			return res;
		}

		template<class Array>
		__host__ __device__ size_t config_to_ordinal_impl(Array const& config) const {
			assert(static_cast<size_t>(config.size()) >= this->sysSize());
			size_t res  = 0;
			size_t base = 1;
			for(size_t l = 0; l != this->sysSize(); ++l, base *= this->dimLoc()) {
				res += config(l) * base;
			}
			return res;
		}

		template<typename... Args>
		__host__ __device__ size_t translate_impl(size_t stateNum, int trans, Args...) const {
			assert(stateNum < this->dim());
			assert(0 <= trans && static_cast<size_t>(trans) < this->sysSize());
			size_t base = 1;
			for(auto l = 0; l != trans; ++l) base *= this->dimLoc();
			size_t const baseCompl = this->dim() / base;
			return stateNum / baseCompl + (stateNum % baseCompl) * base;
		}

		__host__ __device__ size_t reverse_impl(size_t stateNum) const {
			assert(stateNum < this->dim());
			size_t res = 0, base = 1;
			for(size_t l = 0; l != this->sysSize(); ++l, base *= this->dimLoc()) {
				res += (this->dim() / base / this->dimLoc()) * ((stateNum / base) % this->dimLoc());
			}
			return res;
		}
		/* @} */
};