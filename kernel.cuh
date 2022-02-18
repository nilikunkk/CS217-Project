#ifndef GPU_GRAPH_ALGORITHMS_KERNELS_CUH
#define GPU_GRAPH_ALGORITHMS_KERNELS_CUH
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <stdio.h>
#include <string>
#include <string.h>
#include <ctime>

using std::cout;
using std::endl;

//Version 1
__global__ void edgeRelaxationVersion1(int N, int MAX_VAL, int *d_ArrayOfVectices, int *d_ArrayOfIndex, int *d_ArrayOfEdgess, int *d_ArrayOfWeights, int *d_Distance, int *d_DistanceTrack);
__global__ void updateDistanceVersion1(int N, int *d_ArrayOfVectices, int *d_ArrayOfIndex, int *d_ArrayOfEdgess, int *d_ArrayOfWeights, int *d_Distance, int *d_DistanceTrack);
__global__ void updateEdgesIndexVersion1(int N, int *d_ArrayOfVectices, int *d_ArrayOfEdgess, int l, int r);
__global__ void updatePredecessorVersion1(int N, int *d_ArrayOfVectices, int *d_ArrayOfIndex, int *d_ArrayOfEdgess, int *d_ArrayOfWeights, int *d_Distance, int *d_Parents);
__global__ void initializeArrayVersion1(const int N, int *p, const int val, bool sourceDifferent, const int source, const int sourceVal);

//Version 2
__global__ void edgeRelaxationVersion2(int N, int MAX_VAL, int *d_ArrayOfVectices, int *d_ArrayOfIndex, int *d_ArrayOfEdgess, int *d_ArrayOfWeights, int *d_Distance, int *d_DistanceTrack);
__global__ void updateDistanceVersion2(int N, int *d_ArrayOfVectices, int *d_ArrayOfIndex, int *d_ArrayOfEdgess, int *d_ArrayOfWeights, int *d_Distance, int *d_DistanceTrack);
__global__ void updateEdgesIndexVersion2(int N, int *d_ArrayOfVectices, int *d_ArrayOfEdgess, int l, int r);
__global__ void initializeArrayVersion2(const int N, int *p, const int val, bool sourceDifferent, const int source, const int sourceVal);

//Version 3
__global__ void initializeBooleanArrayWithGridStride(const int N, bool *p, const int val, bool sourceDifferent, const int source, const bool sourceVal);
__global__ void edgeRelaxationVersion3(int N, int MAX_VAL, int *d_ArrayOfVectices, int *d_ArrayOfIndex, int *d_ArrayOfEdgess, int *d_ArrayOfWeights, int *d_Distance, int *d_DistanceTrack, bool *p_Flag);
__global__ void updateDistanceVersion3(int N, int *d_ArrayOfVectices, int *d_ArrayOfIndex, int *d_ArrayOfEdgess, int *d_ArrayOfWeights, int *d_Distance, int *d_DistanceTrack, bool *p_Flag);
__global__ void updatePredecessorVersion3(int N, int *d_ArrayOfVectices, int *d_ArrayOfIndex, int *d_ArrayOfEdgess, int *d_ArrayOfWeights, int *d_Distance, int *d_Parents);

#endif //GPU_GRAPH_ALGORITHMS_KERNELS_CUH
