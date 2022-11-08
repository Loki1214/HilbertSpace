#include <catch2/catch_test_macros.hpp>
#include "BaseConverter.hpp"
#include <iostream>
#include <random>

template<typename Integer>
void test_BaseConverter(BaseConverter<Integer> const& bConv, Integer dLoc, Integer L, Integer max) {
	REQUIRE(dLoc == bConv.radix());
	REQUIRE(L == bConv.length());
	REQUIRE(max == bConv.max());

	std::random_device                  seed_gen;
	std::default_random_engine          engine(seed_gen());
	std::uniform_int_distribution<Size> dist(0, bConv.max() - 1);
	constexpr Size                      nSample = 100000;
	Eigen::ArrayX<Size>                 index;
	if(nSample >= bConv.max()) {
		index.resize(bConv.max());
		for(Size j = 0; j != bConv.max(); ++j) index(j) = j;
	}
	else {
		index = index.NullaryExpr(nSample, [&]() { return dist(engine); });
	}

#pragma omp parallel for
	for(Size sample = 0; sample != static_cast<Size>(index.size()); ++sample) {
		auto j      = index(sample);
		auto config = bConv.number_to_config(j);
		REQUIRE(j == bConv.config_to_number(config));
		for(Size pos = 0; pos != bConv.length(); ++pos) {
			REQUIRE(config(pos) == bConv.digit(j, pos));
		}
	}
}

TEST_CASE("BaseConverter", "test") {
	auto powi = [](Size radix, Size expo) {
		Size res = 1;
		for(Size j = 0; j != expo; ++j) res *= radix;
		return res;
	};

	constexpr Size LMax = 20;

	{
		// test Default constructor
		BaseConverter<Size> bConv;
		test_BaseConverter(bConv, Size(0), Size(0), Size(1));
	}
	{
		Size dLoc = 2;
		for(Size L = 0; L <= LMax; ++L) {
			std::cout << "dloc = " << dLoc << ", L = " << L << std::endl;
			BaseConverter<Size> bConv(dLoc, L);
			test_BaseConverter(bConv, dLoc, L, powi(dLoc, L));
		}
	}
	{
		Size dLoc = 4;
		for(Size L = 0; L <= LMax; ++L) {
			std::cout << "dloc = " << dLoc << ", L = " << L << std::endl;
			BaseConverter<Size> bConv(dLoc, L);
			test_BaseConverter(bConv, dLoc, L, powi(dLoc, L));
		}
	}
}