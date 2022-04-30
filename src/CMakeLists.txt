cmake_minimum_required(VERSION 2.8.12)
project(mylib)

set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -I ../../eigen ")

add_subdirectory(pybind11)
pybind11_add_module(mylib main_pybind.cpp)
