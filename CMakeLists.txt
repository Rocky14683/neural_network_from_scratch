cmake_minimum_required(VERSION 3.28)
project(nn_from_scratch)

set(CMAKE_CXX_STANDARD 23)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)


include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(include)

add_executable(nn_from_scratch src/main.cpp)

target_link_libraries(nn_from_scratch
    PUBLIC
        Eigen3::Eigen
)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB TEST_SOURCES test/*.cpp)
add_executable(GTEST ${TEST_SOURCES})

target_include_directories(GTEST PRIVATE include)
target_link_libraries(GTEST gtest gtest_main)

include(GoogleTest)
gtest_discover_tests(GTEST)