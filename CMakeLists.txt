cmake_minimum_required(VERSION 3.20)

project(ImageFiltering LANGUAGES CXX)

find_package(OpenCL REQUIRED)

add_executable(ImageFilteringMain
  src/main.cpp
  kernels/main.cl
)

target_link_libraries(ImageFilteringMain PRIVATE
  OpenCL::OpenCL
)