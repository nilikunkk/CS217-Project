cmake_minimum_required(VERSION 3.14)
project(gpu_graph_algorithms LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)


add_executable(bellman		kernel.cu   kernel.cuh   bellman.cu 
							bellman.cpp 
							main.cpp main.h 
							utilities.cpp utilities.h)

#add_executable(bellman		parallel/kernels.cu parallel/kernels.cuh parallel/bellman.cuh parallel/bellman.cu 
							#sequential/bellman.cpp main.cpp main.h  utilities/utilities.cpp utilities/utilities.h)