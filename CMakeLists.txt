cmake_minimum_required(VERSION 3.20)

project(ImageFiltering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(ImageFilteringMain
  include/clutils.hpp
  include/cvutils.hpp
  src/main.cpp
  kernels/main.cl
)

target_include_directories(ImageFilteringMain PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ImageFilteringMain PRIVATE
  OpenCL::OpenCL
  ${OpenCV_LIBS}
)