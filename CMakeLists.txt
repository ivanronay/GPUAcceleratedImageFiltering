cmake_minimum_required(VERSION 3.20)

project(ImageFiltering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

add_executable(ImageFilteringMain
  src/main.cpp
  kernels/main.cl
)

target_link_libraries(ImageFilteringMain PRIVATE
  OpenCL::OpenCL
)