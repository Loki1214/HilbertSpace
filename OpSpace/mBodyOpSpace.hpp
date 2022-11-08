#pragma once

#include "typedefs.hpp"
#include "ManyBodyOpSpaceBase.hpp"
#include "ManyBodyHilbertSpace/ManyBodySpinSpace.hpp"
#include "OpSpace/OpSpace.hpp"
#include "Algorithm/BaseConverter.hpp"
#include "Algorithm/IntegerComposition.hpp"
#include <Eigen/Dense>

template<class BaseSpace_, typename Scalar_>
class mBodyOpSpace;
template<class BaseSpace_, typename Scalar_>
struct OpSpaceTraits< mBodyOpSpace<BaseSpace_, Scalar_> > {
		using BaseSpace = BaseSpace_;
		using Scalar    = Scalar_;
};
template<class BaseSpace_, typename Scalar_>
struct ManyBodySpaceTraits< mBodyOpSpace<BaseSpace_, Scalar_> > {
		using LocalSpace = OpSpace<Scalar_>;
};

template<typename Scalar_>
class mBodyOpSpace<ManyBodySpinSpace, Scalar_>
    : public ManyBodyOpSpaceBase< mBodyOpSpace<ManyBodySpinSpace, Scalar_> > {
	private:
		using Self = mBodyOpSpace<ManyBodySpinSpace, Scalar_>;
		using Base = ManyBodyOpSpaceBase<Self>;

	public:
		using BaseSpace  = typename Base::BaseSpace;
		using Scalar     = typename Base::Scalar;
		using RealScalar = typename Base::RealScalar;
		using LocalSpace = typename Base::LocalSpace;

	private:
		Size                m_mBody = 0;
		IntegerComposition  m_actingSites;
		BaseConverter<Size> m_opConfig;

	public:
		/**
		 * @brief Custom constructor
		 *
		 * @param m
		 * @param baseSpace
		 */
		__host__ __device__ mBodyOpSpace(Size m, BaseSpace const& baseSpace)
		    : Base(baseSpace, baseSpace.sysSize(), OpSpace<Scalar>(baseSpace.locSpace())),
		      m_mBody{m},
		      m_actingSites(m, baseSpace.sysSize(), 1),
		      m_opConfig(baseSpace.dimLoc() * baseSpace.dimLoc() - 1, m) {}

		mBodyOpSpace()                               = default;
		mBodyOpSpace(mBodyOpSpace const&)            = default;
		mBodyOpSpace& operator=(mBodyOpSpace const&) = default;
		mBodyOpSpace(mBodyOpSpace&&)                 = default;
		mBodyOpSpace& operator=(mBodyOpSpace&&)      = default;
		~mBodyOpSpace()                              = default;

		__host__ __device__ Size m() const { return m_mBody; }

	private:
		/*! @name Implementation for methods of ancestor class HilbertSpace */
		/* @{ */
		friend HilbertSpace< mBodyOpSpace >;
		__host__ __device__ Size dim_impl() const { return m_actingSites.dim() * m_opConfig.max(); }
		/* @} */

		/*! @name Implementation for methods of ancestor class OpSpaceBase */
		/* @{ */
		friend OpSpaceBase< mBodyOpSpace >;
		template<class Array>
		__host__ __device__ void action_impl(Size& resStateNum, Scalar& coeff, Size opNum,
		                                     Size stateNum, Array& work) const;

		__host__ __device__ void action_impl(Size& resStateNum, Scalar& coeff, Size opNum,
		                                     Size stateNum) const {
			Eigen::ArrayX<Size> work(this->sysSize());
			return this->action_impl(resStateNum, coeff, opNum, stateNum, work);
		}

		__host__ __device__ RealScalar opHSNormSq_impl(Size opNum) const {
			assert(opNum < this->dim());
			auto const opConfNum = opNum % m_opConfig.max();
			Size       res       = 1;
			for(Size m = 0; m != this->m(); ++m) {
				res *= this->locSpace().opHSNormSq(1 + m_opConfig.digit(opConfNum, m));
			}
			for(Size l = 0; l != this->sysSize() - this->m(); ++l) {
				res *= this->locSpace().opHSNormSq(0);
			}
			return res;
		}
		/* @} */

		/*! @name Implementation for methods of ancestor class ManyBodySpaceBase */
		/* @{ */
		friend ManyBodySpaceBase< mBodyOpSpace >;
		__host__ __device__ Size locState_impl(Size opNum, int pos) const {
			assert(opNum < this->dim());
			assert(0 <= pos && pos < this->sysSize());
			auto const posConfNum = opNum / m_opConfig.max();
			if(this->m_actingSites.locNumber(posConfNum, pos) == 0)
				return 0;
			else {
				int index = -1;
				for(int l = 0; l <= pos; ++l) {
					index += this->m_actingSites.locNumber(posConfNum, l);
				}
				auto const opConfNum = opNum % m_opConfig.max();
				return this->m_opConfig.digit(opConfNum, index) + 1;
			}
		}
		__host__ __device__ Eigen::RowVectorX<Size> ordinal_to_config_impl(Size opNum) const {
			assert(opNum < this->dim());
			auto const              posConfNum = opNum / m_opConfig.max();
			auto const              opConfNum  = opNum % m_opConfig.max();
			Eigen::RowVectorX<Size> config(this->sysSize());
			this->m_actingSites.ordinal_to_config(config, posConfNum);
			int index = 0;
			for(Size pos = 0; pos != this->sysSize(); ++pos) {
				if(config[pos] == 0) continue;
				config[pos] = 1 + this->m_opConfig.digit(opConfNum, index++);
			}
			return config;
		}
		template<class Array>
		__host__ __device__ Size config_to_ordinal_impl(Array const& config) const {
			assert(static_cast<Size>(config.size()) >= this->sysSize());
			Eigen::ArrayXi posConf   = Eigen::ArrayXi::Zero(this->sysSize());
			Size           opConfNum = 0, base = 1;
			for(Size pos = 0; pos != this->sysSize(); ++pos) {
				if(config[pos] == 0) continue;
				opConfNum += (config[pos] - 1) * base;
				base *= this->m_opConfig.radix();
				posConf[pos] = 1;
			}
			return opConfNum + this->m_actingSites.config_to_ordinal(posConf) * m_opConfig.max();
		}

		template<class Array>
		__host__ __device__ Size translate_impl(Size opNum, int trans, Array& work) const {
			assert(opNum < this->dim());
			assert(0 <= trans && trans < this->sysSize());
			assert(static_cast<Size>(work.size()) >= this->sysSize() + trans);
			work.tail(this->sysSize()) = this->ordinal_to_config(opNum);
			work.head(trans)           = work.tail(trans);
			return this->config_to_ordinal(work);
		}
		__host__ __device__ Size translate_impl(Size opNum, int trans) const {
			assert(opNum < this->dim());
			assert(0 <= trans && trans < this->sysSize());
			Eigen::ArrayX<Size> work(this->sysSize() + trans);
			return this->translate(opNum, trans, work);
		}

		__host__ __device__ Size reverse_impl(Size opNum) const {
			assert(opNum < this->dim());
			auto config = this->ordinal_to_config(opNum);
			for(Size l = 0; l != this->sysSize() / 2; ++l) {
				std::swap(config(l), config(this->sysSize() - 1 - l));
			}
			return this->config_to_ordinal(config);
		}
		/* @} */
};

template<typename Scalar_>
template<class Array>
__host__ __device__ inline void mBodyOpSpace<ManyBodySpinSpace, Scalar_>::action_impl(
    Size& resStateNum, Scalar& coeff, Size opNum, Size stateNum, Array& work) const {
	assert(opNum < this->dim());
	assert(stateNum < this->baseDim());
	assert(static_cast<Size>(work.size()) >= this->sysSize());

	auto const posConfNum = opNum / m_opConfig.max();
	auto const opConfNum  = opNum % m_opConfig.max();
	this->m_actingSites.ordinal_to_config(work, posConfNum);

	resStateNum = stateNum;
	coeff       = 1.0;

	Size   resLocStateNum;
	Scalar resLocCoeff;
	Size   base = 1, locOpId = 0;
	for(Size pos = 0; pos != this->sysSize(); ++pos, base *= this->baseSpace().dimLoc()) {
		auto const locStateNum = this->baseSpace().locState(stateNum, pos);
		// locOpNum = 0 is assumed to corresponds to the identity operator on a site
		if(work[pos] == 0) continue;
		auto const locOpNum = 1 + this->m_opConfig.digit(opConfNum, locOpId++);
		this->locSpace().action(resLocStateNum, resLocCoeff, locOpNum, locStateNum);
		coeff *= resLocCoeff;
		resStateNum += (static_cast<int>(resLocStateNum) - static_cast<int>(locStateNum)) * base;
	}
	assert(resStateNum < this->baseDim());
	return;
}