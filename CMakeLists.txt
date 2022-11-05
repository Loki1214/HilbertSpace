cmake_minimum_required(VERSION 3.14)
project(HilbertSpace CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
find_package(Catch2 QUIET)
if(NOT Catch2_FOUND)
	FetchContent_Declare(catch2
		GIT_REPOSITORY https://github.com/catchorg/Catch2.git
		GIT_TAG v3.1.0
	)
	FetchContent_MakeAvailable(Catch2)
endif()
enable_testing()

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

set(modules
	HilbertSpace
	RandomMatrixEnsemble
)
add_subdirectory(Algorithm)
add_subdirectory(ManyBodyHilbertSpace)
add_subdirectory(OpSpace)

foreach(name IN LISTS modules)
	add_executable(${name}_test ${name}_test.cpp)
	target_compile_options(${name}_test PRIVATE -Wall -Wextra)
	target_link_libraries(${name}_test
		Catch2::Catch2WithMain
		Eigen3::Eigen
		OpenMP::OpenMP_CXX
	)
	add_test(${name}_test ${name}_test)
endforeach(name)