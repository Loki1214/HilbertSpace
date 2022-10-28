#include <catch2/catch_test_macros.hpp>
#include "ManyBodyHilbertSpace.hpp"
#include <random>
#include <iostream>

TEST_CASE("ManyBodyHilbertSpace", "test") {
	size_t dLoc = 2;
	HilbertSpace<int> locSpace(dLoc);

	size_t sysSize = 4;
	ManyBodySpinSpace mbSpace(sysSize, locSpace);
	REQUIRE(mbSpace.dim() == 16);
}