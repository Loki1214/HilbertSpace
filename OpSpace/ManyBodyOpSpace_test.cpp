#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include "../ManyBodySpaceBase_test.hpp"
#include "../OpSpaceBase_test.hpp"
#include "ManyBodyOpSpace.hpp"
#include "../ManyBodyHilbertSpace/ManyBodySpinSpace.hpp"
#include <iostream>
#include <complex>

using Scalar = std::complex<double>;

TEST_CASE("ManyBodyOpSpace", "test") {
	size_t const      LMax = 15;
	size_t const      dLoc = 2;
	HilbertSpace<int> locSpace(dLoc);

	// test for class ManyBodySpinSpace
	{
		// Default constructor
		ManyBodySpinSpace                          mbSpace;
		ManyBodyOpSpace<decltype(mbSpace), Scalar> opSpace;
		test_ManyBodySpaceBase(opSpace, 0, opSpace.locSpace());
		// test_OpSpace(opSpace);
	}
	{
		// test Constructor1
		ManyBodySpinSpace                          mbSpace(0, locSpace);
		ManyBodyOpSpace<decltype(mbSpace), Scalar> opSpace(mbSpace);
		test_ManyBodySpaceBase(opSpace, 0, OpSpace<Scalar>(locSpace));
		// test_OpSpace(opSpace);
		for(size_t sysSize = LMax; sysSize <= LMax; ++sysSize) {
			std::cout << "sysSize = " << sysSize << std::endl;
			ManyBodySpinSpace                          mbSpace(sysSize, locSpace);
			ManyBodyOpSpace<decltype(mbSpace), Scalar> opSpace(mbSpace);
			test_ManyBodySpaceBase(opSpace, sysSize, OpSpace<Scalar>(locSpace));
			// test_OpSpace(opSpace);
		}
	}
}