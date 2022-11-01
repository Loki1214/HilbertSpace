#include <catch2/catch_test_macros.hpp>
#include "../ManyBodySpaceBase_test.hpp"
#include "../OpSpaceBase_test.hpp"
#include "mBodyOpSpace.hpp"
#include "../ManyBodyHilbertSpace/ManyBodySpinSpace.hpp"
#include <iostream>
#include <complex>

using Scalar = std::complex<double>;

TEST_CASE("ManyBodyOpSpace", "test") {
	size_t const      LMax = 16;
	size_t const      dLoc = 2;
	HilbertSpace<int> locSpace(dLoc);

	// test for class ManyBodySpinSpace
	{
		// Default constructor
		ManyBodySpinSpace                       mbSpace;
		mBodyOpSpace<decltype(mbSpace), Scalar> opSpace;
		test_ManyBodySpaceBase(opSpace, 0, opSpace.locSpace());
		test_OpSpace(opSpace);
	}
	{
		// test Constructor1
		ManyBodySpinSpace                       mbSpace(0, locSpace);
		mBodyOpSpace<decltype(mbSpace), Scalar> opSpace(0, mbSpace);
		test_ManyBodySpaceBase(opSpace, 0, OpSpace<Scalar>(locSpace));
		test_OpSpace(opSpace);
		for(size_t sysSize = 1; sysSize <= LMax; ++sysSize) {
			ManyBodySpinSpace mbSpace(sysSize, locSpace);
			for(size_t m = 1; m <= sysSize; ++m) {
				std::cout << "sysSize = " << sysSize << ", m = " << m << std::endl;
				mBodyOpSpace<decltype(mbSpace), Scalar> opSpace(m, mbSpace);
				if(m > 4 && opSpace.dim() > 100000000) {
					std::cout << "opSpace.dim() = " << opSpace.dim()
					          << " is so large. Skipping test..." << std::endl;
					continue;
				}
				test_ManyBodySpaceBase(opSpace, sysSize, OpSpace<Scalar>(locSpace));
				test_OpSpace(opSpace);
			}
		}
	}
}