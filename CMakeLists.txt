cmake_minimum_required(VERSION 3.14)
project(HilbertSpace CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(catch2
	GIT_REPOSITORY https://github.com/catchorg/Catch2.git
	GIT_TAG v3.1.0
)
FetchContent_MakeAvailable(Catch2)
enable_testing()

set(modules
	HilbertSpace
)

foreach(name IN LISTS modules)
	add_executable(${name}_test ${name}_test.cpp)
	target_compile_options(${name}_test PRIVATE -Wall -Wextra)
	target_link_libraries(${name}_test Catch2::Catch2WithMain)
	add_test(${name}_test ${name}_test)
endforeach()