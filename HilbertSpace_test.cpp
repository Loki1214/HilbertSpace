#include <catch2/catch_test_macros.hpp>
#include "HilbertSpace.hpp"

TEST_CASE("HilbertSpace", "test") {
	HilbertSpace<int> hSpace;
	REQUIRE(hSpace.dim() == 0);
}