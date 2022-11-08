#include <catch2/catch_test_macros.hpp>
#include "HilbertSpace.hpp"
#include <random>
#include <iostream>

TEST_CASE("HilbertSpace", "test") {
	std::random_device              seed_gen;
	std::default_random_engine      engine(seed_gen());
	std::uniform_int_distribution<> dist(0, 100000);

	Size testLoop = 100;

	{
		// Default constructor
		{
			HilbertSpace<int> hSpace;
			REQUIRE(hSpace.dim() == 0);
		}
		for(Size n = 0; n != testLoop; ++n) {
			Size              dim = dist(engine);
			HilbertSpace<int> hSpace(dim);
			REQUIRE(hSpace.dim() == dim);
		}
	}
	{
		// Copy constructor
		for(Size n = 0; n != testLoop; ++n) {
			Size              dim = dist(engine);
			HilbertSpace<int> hSpace1(dim);
			HilbertSpace<int> hSpace2(hSpace1);
			REQUIRE(hSpace1.dim() == hSpace2.dim());
		}
	}
	{
		// Move constructor
		for(Size n = 0; n != testLoop; ++n) {
			Size              dim = dist(engine);
			HilbertSpace<int> hSpace1(dim);
			HilbertSpace<int> hSpace2(std::move(hSpace1));
			REQUIRE(hSpace2.dim() == dim);
		}
	}

	{
		// Copy assignment operator
		for(Size n = 0; n != testLoop; ++n) {
			auto              dim = dist(engine);
			HilbertSpace<int> hSpace1(dim);
			HilbertSpace<int> hSpace2;
			hSpace2 = hSpace1;
			REQUIRE(hSpace1.dim() == hSpace2.dim());
		}
	}
	{
		// Move assignment operator
		for(Size n = 0; n != testLoop; ++n) {
			Size              dim = dist(engine);
			HilbertSpace<int> hSpace1(dim);
			HilbertSpace<int> hSpace2;
			hSpace2 = std::move(hSpace1);
			REQUIRE(hSpace2.dim() == dim);
		}
	}

	{
		// Equality operator
		for(Size n = 0; n != testLoop; ++n) {
			Size              dim = dist(engine);
			HilbertSpace<int> hSpace1(dim);
			HilbertSpace<int> hSpace2(hSpace1);
			REQUIRE(hSpace1.dim() == hSpace2.dim());
			REQUIRE(hSpace1 == hSpace2);
		}
	}
}