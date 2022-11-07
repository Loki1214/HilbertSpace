#pragma once

#include "../ManyBodyOpSpaceBase.hpp"
#include "OpSpace.hpp"
#include <Eigen/Dense>

template<class BaseSpace_, typename Scalar_>
class ManyBodyOpSpace;
template<class BaseSpace_, typename Scalar_>
struct OpSpaceTraits< ManyBodyOpSpace<BaseSpace_, Scalar_> > {
		using BaseSpace = BaseSpace_;
		using Scalar    = Scalar_;
};
template<class BaseSpace_, typename Scalar_>
struct ManyBodySpaceTraits< ManyBodyOpSpace<BaseSpace_, Scalar_> > {
		using LocalSpace = OpSpace<Scalar_>;
};

template<class BaseSpace_, typename Scalar_>
class ManyBodyOpSpace : public ManyBodyOpSpaceBase< ManyBodyOpSpace<BaseSpace_, Scalar_> > {
	private:
		using Self = ManyBodyOpSpace<BaseSpace_, Scalar_>;
		using Base = ManyBodyOpSpaceBase<Self>;

	public:
		using BaseSpace  = typename Base::BaseSpace;
		using Scalar     = typename Base::Scalar;
		using RealScalar = typename Base::RealScalar;
		using LocalSpace = typename Base::LocalSpace;

	public:
		/**
		 * @brief Custom constructor
		 *
		 * @param baseSpace
		 */
		__host__ __device__ ManyBodyOpSpace(BaseSpace const& baseSpace)
		    : Base(baseSpace, baseSpace.sysSize(), LocalSpace(baseSpace.locSpace())){};
		__host__ __device__ ManyBodyOpSpace(BaseSpace&& baseSpace)
		    : Base(std::move(baseSpace), baseSpace.sysSize(), LocalSpace(baseSpace.locSpace())){};

		ManyBodyOpSpace()                                  = default;
		ManyBodyOpSpace(ManyBodyOpSpace const&)            = default;
		ManyBodyOpSpace& operator=(ManyBodyOpSpace const&) = default;
		ManyBodyOpSpace(ManyBodyOpSpace&&)                 = default;
		ManyBodyOpSpace& operator=(ManyBodyOpSpace&&)      = default;
		~ManyBodyOpSpace()                                 = default;

	private:
		/*! @name Implementation for methods of ancestor class HilbertSpace */
		/* @{ */
		friend HilbertSpace<ManyBodyOpSpace>;
		__host__ __device__ size_t dim_impl() const { return this->baseDim() * this->baseDim(); }
		/* @} */

		/*! @name Implementation for methods of ancestor class OpSpaceBase */
		/* @{ */
		friend OpSpaceBase< ManyBodyOpSpace >;
		__host__ __device__ void action_impl(size_t& resStateNum, Scalar& coeff, size_t opNum,
		                                     size_t basisNum) const {
			resStateNum = basisNum;
			coeff       = {1, 0};
			size_t base = 1;
			for(size_t pos = 0; pos != this->sysSize(); ++pos, base *= this->baseSpace().dimLoc()) {
				auto locOpNum                      = this->locState(opNum, pos);
				auto locBasisNum                   = this->baseSpace().locState(basisNum, pos);
				auto [resLocBasisNum, resLocCoeff] = this->locSpace().action(locOpNum, locBasisNum);
				if(abs(resLocCoeff) == 0) {
					coeff       = 0;
					resStateNum = basisNum;
					break;
				}
				coeff *= resLocCoeff;
				resStateNum
				    += (static_cast<int>(resLocBasisNum) - static_cast<int>(locBasisNum)) * base;
			}
		}

		__host__ __device__ RealScalar opHSNormSq_impl(size_t opNum) const {
			size_t res = 1;
			for(size_t pos = 0; pos != this->sysSize(); ++pos) {
				res *= this->locSpace().opHSNormSq(this->locState(opNum, pos));
			}
			return res;
		}
		/* @} */

		/*! @name Implementation for methods of ancestor class ManyBodySpaceBase */
		/* @{ */
		friend ManyBodySpaceBase< ManyBodyOpSpace >;
		__host__ __device__ size_t locState_impl(size_t stateNum, int pos) const {
			assert(stateNum < this->dim());
			assert(0 <= pos && static_cast<size_t>(pos) < this->sysSize());
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

		__host__ __device__ size_t translate_impl(size_t stateNum, int trans) const {
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