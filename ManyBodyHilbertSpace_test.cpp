#include <catch2/catch_test_macros.hpp>
#include "ManyBodyHilbertSpace.hpp"
#include <random>
#include <iostream>

template<class Derived, class LocalSpace>
void test_ManyBodySpaceBase(ManyBodySpaceBase<Derived> const& mbSpace, size_t sysSize,
                            LocalSpace const& locSpace) {
	auto powi = [](size_t base, size_t n) {
		size_t res = 1;
		for(size_t j = 0; j != n; ++j) res *= base;
		return res;
	};
	if(sysSize == 0)
		REQUIRE(mbSpace.dim() == 0);
	else
		REQUIRE(mbSpace.dim() == powi(locSpace.dim(), sysSize));
	REQUIRE(mbSpace.sysSize() == sysSize);
	REQUIRE(mbSpace.locSpace() == locSpace);
	REQUIRE(mbSpace.dimLoc() == locSpace.dim());

// test locState
// test ordinal_to_config
// test config_to_ordinal
#pragma omp parallel for
	for(size_t stateNum = 0; stateNum != mbSpace.dim(); ++stateNum) {
		auto config = mbSpace.ordinal_to_config(stateNum);
		REQUIRE(stateNum == mbSpace.config_to_ordinal(config));
		for(size_t pos = 0; pos != mbSpace.sysSize(); ++pos) {
			REQUIRE(config(pos) == mbSpace.locState(stateNum, pos));
		}
	}

	// test for translation operations
	mbSpace.compute_transEqClass();
	REQUIRE(mbSpace.transPeriod().sum() == static_cast<int>(mbSpace.dim()));

	Eigen::ArrayXi appeared = Eigen::ArrayXi::Zero(mbSpace.dim());
#pragma omp parallel for
	for(size_t eqClassNum = 0; eqClassNum != mbSpace.transEqDim(); ++eqClassNum) {
		auto stateNum = mbSpace.transEqClassRep(eqClassNum);
		appeared(stateNum) += 1;
		for(auto trans = 1; trans != mbSpace.transPeriod(eqClassNum); ++trans) {
			auto translated = mbSpace.translate(stateNum, trans);
			appeared(translated) += 1;
		}
	}
#pragma omp parallel for
	for(size_t stateNum = 0; stateNum != mbSpace.dim(); ++stateNum)
		REQUIRE(appeared(stateNum) == 1);

		// test for state_to_transEqClass
		// test for state_to_transShift
#pragma omp parallel for
	for(size_t stateNum = 0; stateNum != mbSpace.dim(); ++stateNum) {
		auto const eqClass    = mbSpace.state_to_transEqClass(stateNum);
		auto const eqClassRep = mbSpace.transEqClassRep(eqClass);
		auto const trans      = mbSpace.state_to_transShift(stateNum);
		REQUIRE(static_cast<int>(stateNum) == mbSpace.translate(eqClassRep, trans));
	}

	// test for reverse()
#pragma omp parallel for
	for(size_t stateNum = 0; stateNum != mbSpace.dim(); ++stateNum) {
		auto config   = mbSpace.ordinal_to_config(stateNum);
		auto reversed = mbSpace.config_to_ordinal(config.reverse());
		REQUIRE(static_cast<int>(reversed) == mbSpace.reverse(stateNum));
	}
}

TEST_CASE("ManyBodyHilbertSpace", "test") {
	size_t            dLoc = 2;
	HilbertSpace<int> locSpace(dLoc);

	// test for class ManyBodySpinSpace
	{
		// Default constructor
		ManyBodySpinSpace mbSpace;
		test_ManyBodySpaceBase(mbSpace, 0, HilbertSpace<int>());
	}
	{
		// test Constructor1
		ManyBodySpinSpace mbSpace(0, locSpace);
		test_ManyBodySpaceBase(mbSpace, 0, locSpace);
		for(size_t sysSize = 1; sysSize <= 20; ++sysSize) {
			ManyBodySpinSpace mbSpace(sysSize, locSpace);
			test_ManyBodySpaceBase(mbSpace, sysSize, locSpace);
		}
	}
}