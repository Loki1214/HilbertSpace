#include <catch2/catch_test_macros.hpp>
#include "BaseConverter.hpp"

template<typename Integer>
void test_BaseConverter(BaseConverter<Integer> const& bConv, Integer dLoc, Integer L, Integer max) {
	REQUIRE(dLoc == bConv.radix());
	REQUIRE(L == bConv.length());
	REQUIRE(max == bConv.max());
	for(size_t j = 0; j != bConv.max(); ++j) {
		auto config = bConv.number_to_config(j);
		REQUIRE(j == bConv.config_to_number(config));
		for(size_t pos = 0; pos != bConv.length(); ++pos) {
			REQUIRE(config(pos) == bConv.digit(j, pos));
		}
	}
}

TEST_CASE("BaseConverter", "test") {
	auto powi = [](size_t radix, size_t expo) {
		size_t res = 1;
		for(size_t j = 0; j != expo; ++j) res *= radix;
		return res;
	};

	constexpr size_t LMax = 20;

	{
		// test Default constructor
		BaseConverter<size_t> bConv;
		test_BaseConverter(bConv, size_t(0), size_t(0), size_t(1));
	}
	{
		size_t dLoc = 2;
		for(size_t L = 0; L <= LMax; ++L) {
			BaseConverter<size_t> bConv(dLoc, L);
			test_BaseConverter(bConv, dLoc, L, powi(dLoc, L));
		}
	}
}