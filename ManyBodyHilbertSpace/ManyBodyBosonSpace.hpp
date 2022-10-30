#pragma once
#include "../ManyBodySpaceBase.hpp"
#include "../Algorithm/IntegerComposition.hpp"

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
		 * @brief Custom constructor 1
		 *
		 * @param sysSize
		 * @param max
		 * @param locSpace
		 */
		__host__ __device__ ManyBodyBosonSpace(size_t sysSize, size_t max,
		                                       LocalSpace const& locSpace)
		    : Base(sysSize, locSpace),
		      m_iComp{(locSpace.dim() == 0 ? 0 : locSpace.dim() - 1), sysSize, max} {}

		/**
		 * @brief Custom constructor 2
		 *
		 * @param sysSize
		 * @param locSpace
		 */
		__host__ __device__ ManyBodyBosonSpace(size_t sysSize, LocalSpace const& locSpace)
		    : ManyBodyBosonSpace(sysSize, locSpace.dim(), locSpace) {}

		/**
		 * @brief Custom constructor 3
		 *
		 * @param sysSize
		 * @param locSpace
		 */
		__host__ __device__ ManyBodyBosonSpace(size_t sysSize, size_t nBosons, size_t max)
		    : ManyBodyBosonSpace(sysSize, max, LocalSpace(nBosons + 1)) {}

		/**
		 * @brief Default constructor
		 *
		 * @param sysSize
		 * @param nBosons
		 */
		__host__ __device__ ManyBodyBosonSpace(size_t sysSize = 0, size_t nBosons = 0)
		    : ManyBodyBosonSpace(sysSize, LocalSpace(sysSize == 0 ? 0 : nBosons + 1)) {}

	private:
		/*! @name Implementation for methods of ancestor class HilbertSpace */
		/* @{ */
		friend HilbertSpace<ManyBodyBosonSpace>;
		__host__ __device__ size_t dim_impl() const { return m_iComp.dim(); }
		/* @} */

		/*! @name Implementation for methods of parent class ManyBodySpaceBase */
		/* @{ */
		friend ManyBodySpaceBase<ManyBodyBosonSpace>;
		__host__ __device__ size_t locState_impl(size_t stateNum, int pos) const {
			assert(stateNum < this->dim());
			assert(0 <= pos && static_cast<size_t>(pos) < this->sysSize());
			return m_iComp.locNumber(stateNum, pos);
		}

		__host__ __device__ Eigen::RowVectorX<size_t> ordinal_to_config_impl(
		    size_t stateNum) const {
			assert(stateNum < this->dim());
			return m_iComp.ordinal_to_config(stateNum);
		}

		template<class Array>
		__host__ __device__ size_t config_to_ordinal_impl(Array const& config) const {
			assert(static_cast<size_t>(config.size()) >= this->sysSize());
			return m_iComp.config_to_ordinal(config);
		}

		template<typename... Args>
		__host__ __device__ size_t translate_impl(size_t stateNum, int trans, Args...) const {
			assert(stateNum < this->dim());
			assert(0 <= trans && static_cast<size_t>(trans) < this->sysSize());

			Eigen::ArrayX<size_t> config(this->sysSize() + trans);
			config.tail(this->sysSize()) = m_iComp.ordinal_to_config(stateNum);
			config.head(trans)           = config.tail(trans);
			return m_iComp.config_to_ordinal(config);
		}

		__host__ __device__ size_t reverse_impl(size_t stateNum) const {
			assert(stateNum < this->dim());
			auto config = m_iComp.ordinal_to_config(stateNum);
			for(size_t l = 0; l != this->sysSize() / 2; ++l) {
				auto temp                       = config(l);
				config(l)                       = config(this->sysSize() - 1 - l);
				config(this->sysSize() - 1 - l) = temp;
			}
			return m_iComp.config_to_ordinal(config);
		}
		/* @} */
};