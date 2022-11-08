#pragma once

#include "typedefs.hpp"
#include "ManyBodySpaceBase.hpp"
#include "Algorithm/IntegerComposition.hpp"

class ManyBodyBosonSpace;
template<>
struct ManyBodySpaceTraits<ManyBodyBosonSpace> {
		using LocalSpace = HilbertSpace<int>;
};

class ManyBodyBosonSpace : public ManyBodySpaceBase<ManyBodyBosonSpace> {
	private:
		using Base = ManyBodySpaceBase<ManyBodyBosonSpace>;

	public:
		using LocalSpace = typename Base::LocalSpace;

	private:
		IntegerComposition m_iComp;

	public:
		/**
		 * @brief Custom constructor 1
		 *
		 * @param sysSize
		 * @param max
		 * @param locSpace
		 */
		__host__ __device__ ManyBodyBosonSpace(Size sysSize, Size max,
		                                       LocalSpace const& locSpace)
		    : Base(sysSize, locSpace),
		      m_iComp{(locSpace.dim() == 0 ? 0 : locSpace.dim() - 1), sysSize, max} {}

		/**
		 * @brief Custom constructor 2
		 *
		 * @param sysSize
		 * @param locSpace
		 */
		__host__ __device__ ManyBodyBosonSpace(Size sysSize, LocalSpace const& locSpace)
		    : ManyBodyBosonSpace(sysSize, locSpace.dim(), locSpace) {}

		/**
		 * @brief Custom constructor 3
		 *
		 * @param sysSize
		 * @param locSpace
		 */
		__host__ __device__ ManyBodyBosonSpace(Size sysSize, Size nBosons, Size max)
		    : ManyBodyBosonSpace(sysSize, max, LocalSpace(nBosons + 1)) {}

		/**
		 * @brief Default constructor
		 *
		 * @param sysSize
		 * @param nBosons
		 */
		__host__ __device__ ManyBodyBosonSpace(Size sysSize = 0, Size nBosons = 0)
		    : ManyBodyBosonSpace(sysSize, LocalSpace(sysSize == 0 ? 0 : nBosons + 1)) {}

	private:
		/*! @name Implementation for methods of ancestor class HilbertSpace */
		/* @{ */
		friend HilbertSpace<ManyBodyBosonSpace>;
		__host__ __device__ Size dim_impl() const { return m_iComp.dim(); }
		/* @} */

		/*! @name Implementation for methods of parent class ManyBodySpaceBase */
		/* @{ */
		friend ManyBodySpaceBase<ManyBodyBosonSpace>;
		__host__ __device__ Size locState_impl(Size stateNum, int pos) const {
			assert(stateNum < this->dim());
			assert(0 <= pos && static_cast<Size>(pos) < this->sysSize());
			return m_iComp.locNumber(stateNum, pos);
		}

		__host__ __device__ Eigen::RowVectorX<Size> ordinal_to_config_impl(
		    Size stateNum) const {
			assert(stateNum < this->dim());
			return m_iComp.ordinal_to_config(stateNum);
		}

		template<class Array>
		__host__ __device__ Size config_to_ordinal_impl(Array const& config) const {
			assert(static_cast<Size>(config.size()) >= this->sysSize());
			return m_iComp.config_to_ordinal(config);
		}

		template<typename... Args>
		__host__ __device__ Size translate_impl(Size stateNum, int trans,
		                                          Args&&... args) const {
			assert(stateNum < this->dim());
			assert(0 <= trans && static_cast<Size>(trans) < this->sysSize());
			return m_iComp.translate(stateNum, trans, std::forward<Args>(args)...);
		}

		__host__ __device__ Size reverse_impl(Size stateNum) const {
			assert(stateNum < this->dim());
			auto config = m_iComp.ordinal_to_config(stateNum);
			for(Size l = 0; l != this->sysSize() / 2; ++l) {
				std::swap(config(l), config(this->sysSize() - 1 - l));
			}
			return m_iComp.config_to_ordinal(config);
		}
		/* @} */
};