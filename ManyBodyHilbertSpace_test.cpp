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
	// test ordinalToConfig
	// test configToOrdinal
	for(size_t stateNum = 0; stateNum != mbSpace.dim(); ++stateNum) {
		auto config = mbSpace.ordinalToConfig(stateNum);
		REQUIRE(stateNum == mbSpace.configToOrdinal(config));
		for(size_t pos = 0; pos != mbSpace.sysSize(); ++pos) {
			REQUIRE(config(pos) == mbSpace.locState(stateNum, pos));
		}
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