#include <catch2/catch_test_macros.hpp>
#include "../ManyBodySpaceBase_test.hpp"
#include "ManyBodySpinSpace.hpp"
#include <iostream>

TEST_CASE("ManyBodySpinSpace", "test") {
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