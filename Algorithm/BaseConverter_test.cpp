#include <catch2/catch_test_macros.hpp>
#include "BaseConverter.hpp"
#include <iostream>
#include <random>

template<typename Integer>
void test_BaseConverter(BaseConverter<Integer> const& bConv, Integer dLoc, Integer L, Integer max) {
	REQUIRE(dLoc == bConv.radix());
	REQUIRE(L == bConv.length());
	REQUIRE(max == bConv.max());

	std::random_device                    seed_gen;
	std::default_random_engine            engine(seed_gen());
	std::uniform_int_distribution<size_t> dist(0, bConv.max() - 1);
	constexpr size_t                      nSample = 100000;
	Eigen::ArrayX<size_t>                 index;
	if(nSample >= bConv.max()) {
		index.resize(bConv.max());
		for(size_t j = 0; j != bConv.max(); ++j) index(j) = j;
	}
	else {
		index = index.NullaryExpr(nSample, [&]() { return dist(engine); });
	}

#pragma omp parallel for
	for(size_t sample = 0; sample != static_cast<size_t>(index.size()); ++sample) {
		auto j      = index(sample);
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
			std::cout << "dloc = " << dLoc << ", L = " << L << std::endl;
			BaseConverter<size_t> bConv(dLoc, L);
			test_BaseConverter(bConv, dLoc, L, powi(dLoc, L));
		}
	}
	{
		size_t dLoc = 4;
		for(size_t L = 0; L <= 16; ++L) {
			std::cout << "dloc = " << dLoc << ", L = " << L << std::endl;
			BaseConverter<size_t> bConv(dLoc, L);
			test_BaseConverter(bConv, dLoc, L, powi(dLoc, L));
		}
	}
}