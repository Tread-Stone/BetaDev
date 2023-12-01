cmake_minimum_required(VERSION 3.10)

# Set the project name
project(nn VERSION 1.0)

# Specify C++ standard
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# To use Clang, uncomment the following lines
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

# set -xe
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -fno-omit-frame-pointer -fsanitize=address")

# Add compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

# Add the executable
add_executable(nn
  main.c
  lib/neuralnet.h
)

# Link libraries
target_link_libraries(nn)

