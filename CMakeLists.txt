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

add_executable(HilbertSpace_test HilbertSpace_test.cpp)
target_compile_options(HilbertSpace_test PRIVATE -Wall -Wextra)
target_link_libraries(HilbertSpace_test Catch2::Catch2WithMain)
add_test(HilbertSpace_test HilbertSpace_test)