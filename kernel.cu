#include "kernel.cuh"


__global__ void edgeRelaxationVersion1(int N, int MAX_VAL, int *d_ArrayOfVectices, int *d_ArrayOfIndex, int *d_ArrayOfEdgess, int *d_ArrayOfWeights, int *d_Distance, int *d_DistanceTrack) {
    unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);

    if (index < N - 1) { // do index < N - 1 because nth element of I array points to the end of E array
        for (int j = d_ArrayOfIndex[index]; j < d_ArrayOfIndex[index + 1]; j++) {
            int w = d_ArrayOfWeights[j];
            int du = d_Distance[index];
            int dv = d_Distance[d_ArrayOfEdgess[j]];
            int newDist = du + w;
            // Check if the distance is already set to max then just take the max since,
            // Cuda implementation gives this when a number is added to already max value of int.
            // E.g 2147483647 + 5 becomes -2147483644
            if (du == MAX_VAL){
                newDist = MAX_VAL;
            }

            if (newDist < dv) {
                atomicMin(&d_DistanceTrack[d_ArrayOfEdgess[j]],newDist);
            }
        }
    }
}

__global__ void edgeRelaxationVersion2(int N, int MAX_VAL, int *d_ArrayOfVectices, int *d_ArrayOfIndex, int *d_ArrayOfEdgess, int *d_ArrayOfWeights, int *d_Distance, int *d_DistanceTrack){
    unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    unsigned int stride = blockDim.x * gridDim.x;

    //if(tid ==0){
    //    printf("block dim = %d, grid dim = %d, stride = %d\n",blockDim.x,gridDim.x, stride);
    //}
    for (int index = tid; index < N - 1; index += stride){  // do index < N - 1 because nth element of I array points to the end of E array
        for (int j = d_ArrayOfIndex[index]; j < d_ArrayOfIndex[index + 1]; j++) {
            int w = d_ArrayOfWeights[j];
            int du = d_Distance[index];
            int dv = d_Distance[d_ArrayOfEdgess[j]];
            int newDist = du + w;
            // Check if the distance is already set to max then just take the max since,
            // Cuda implementation gives this when a number is added to already max value of int.
            // E.g 2147483647 + 5 becomes -2147483644
            if (du == MAX_VAL){
                newDist = MAX_VAL;
            }
            //printf("Index = %d, w=%d, du =%d, dv=%d,  -- du + w = %d\n", index, w, du , dv, du + w);

            if (newDist < dv) {
                atomicMin(&d_DistanceTrack[d_ArrayOfEdgess[j]],newDist);
            }
        }
    }
}

__global__ void updateDistanceVersion1(int N, int *d_ArrayOfVectices, int *d_ArrayOfIndex, int *d_ArrayOfEdgess, int *d_ArrayOfWeights, int *d_Distance, int *d_DistanceTrack) {
    unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);
    if (index < N) {

        if (d_Distance[index] > d_DistanceTrack[index]) {
            d_Distance[index] = d_DistanceTrack[index];
        }
        d_DistanceTrack[index] = d_Distance[index];
    }
}

__global__ void updateDistanceVersion2(int N, int *d_ArrayOfVectices, int *d_ArrayOfIndex, int *d_ArrayOfEdgess, int *d_ArrayOfWeights, int *d_Distance, int *d_DistanceTrack) {
    unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    unsigned int stride = blockDim.x * gridDim.x;

    for (int index = tid; index < N ; index += stride) {  // do stride
            if (d_Distance[index] > d_DistanceTrack[index]) {
                d_Distance[index] = d_DistanceTrack[index];
            }
            d_DistanceTrack[index] = d_Distance[index];
    }
}

__global__ void updateEdgesIndexVersion1(int N, int *d_ArrayOfVectices, int *d_ArrayOfEdgess, int l, int r) {
    unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);

    // This does binary search on the V array to find the index of each node in the Edge array (E) and replace the same with index
    // Based on the iterative binary search from : https://www.geeksforgeeks.org/binary-search/
    if (index < N) {
        while (l <= r) {
            int m = l + (r - l) / 2;
            // Check if x is present at mid
            if (d_ArrayOfVectices[m] == d_ArrayOfEdgess[index]) {
                d_ArrayOfEdgess[index] = m;
                break;
            }
            // If x greater, ignore left half
            if (d_ArrayOfVectices[m] < d_ArrayOfEdgess[index]) {
                l = m + 1;
            } else {        // If x is smaller, ignore right half
                r = m - 1;
            }
        }
    }
}

__global__ void updateEdgesIndexVersion2(int N, int *d_ArrayOfVectices, int *d_ArrayOfEdgess, int l, int r) {
    unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    unsigned int stride = blockDim.x * gridDim.x;
    int left = l;
    int right = r;

    for (int index = tid; index < N ; index += stride) {  // do stride
        // This does binary search on the V array to find the index of each node in the Edge array (E) and replace the same with index
        // Based on the iterative binary search from : https://www.geeksforgeeks.org/binary-search/
        left = l;
        right = r; // reset for each stride

        while (left <= right) {
            int m = left + (right - left) / 2;
            // Check if x is present at mid
            if (d_ArrayOfVectices[m] == d_ArrayOfEdgess[index]) {
                d_ArrayOfEdgess[index] = m;
                break;
            }
            // If x greater, ignore left half
            if (d_ArrayOfVectices[m] < d_ArrayOfEdgess[index]) {
                left = m + 1;
            } else {        // If x is smaller, ignore right half
                right = m - 1;
            }
         }
    }
}

__global__ void initializeArrayVersion1(const int N, int *p, const int val, bool sourceDifferent, const int source, const int sourceVal){
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index < N) {
        p[index] = val;
        if(sourceDifferent){
            if(index == source) {
                p[index] = sourceVal;
            }
        }
    }
}

__global__ void initializeArrayVersion2(const int N, int *p, const int val, bool sourceDifferent, const int source, const int sourceVal){

    unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    unsigned int stride = blockDim.x * gridDim.x;

    for (int index = tid; index < N ; index += stride) {  // do stride
        p[index] = val;
        if(sourceDifferent){
            if(index == source) {
                p[index] = sourceVal;
            }
        }
    }
}

__global__ void initializeBooleanArrayWithGridStride(const int N, bool *p, const int val, bool sourceDifferent, const int source, const bool sourceVal){

    unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    unsigned int stride = blockDim.x * gridDim.x;

    for (int index = tid; index < N ; index += stride) {  // do stride
        p[index] = val;
        if(sourceDifferent){
            if(index == source) {
                p[index] = sourceVal;
            }
        }
    }
}

// Only relax the outgoing edges if the vertex has lower distance based on the Flag
__global__ void edgeRelaxationVersion3(int N, int MAX_VAL, int *d_ArrayOfVectices, int *d_ArrayOfIndex, int *d_ArrayOfEdgess, int *d_ArrayOfWeights, int *d_Distance, int *d_DistanceTrack, bool *d_Flag){
    unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    unsigned int stride = blockDim.x * gridDim.x;

    for (int index = tid; index < N - 1; index += stride){  // do index < N - 1 because nth element of I array points to the end of E array
        if (d_Flag[index]) {
            d_Flag[index] = false;
            for (int j = d_ArrayOfIndex[index]; j < d_ArrayOfIndex[index + 1]; j++) {
                int w = d_ArrayOfWeights[j];
                int du = d_Distance[index];
                int dv = d_Distance[d_ArrayOfEdgess[j]];
                int newDist = du + w;
                // Check if the distance is already set to max then just take the max since,
                // Cuda implementation gives this when a number is added to already max value of int.
                // E.g 2147483647 + 5 becomes -2147483644
                if (du == MAX_VAL) {
                    newDist = MAX_VAL;
                }
                //printf("Index = %d, w=%d, du =%d, dv=%d,  -- du + w = %d\n", index, w, du , dv, du + w);

                if (newDist < dv) {
                    atomicMin(&d_DistanceTrack[d_ArrayOfEdgess[j]], newDist);
                }
            }
        }
    }
}

// Sets the flag to true if distance of vertex is changed hence all outgoing edges need to be relaxed in the next round
__global__ void updateDistanceVersion3(int N, int *d_ArrayOfVectices, int *d_ArrayOfIndex, int *d_ArrayOfEdgess, int *d_ArrayOfWeights, int *d_Distance, int *d_DistanceTrack, bool *d_Flag) {
    unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    unsigned int stride = blockDim.x * gridDim.x;

    for (int index = tid; index < N ; index += stride) {  // do stride
        if (d_Distance[index] > d_DistanceTrack[index]) {
            d_Distance[index] = d_DistanceTrack[index];
            d_Flag[index] = true;
        }
        d_DistanceTrack[index] = d_Distance[index];
    }
}

// Update d_Parents when Bellman-Ford algorithm is completed (for version 1)
__global__ void updatePredecessorVersion1(int N, int *d_ArrayOfVectices, int *d_ArrayOfIndex, int *d_ArrayOfEdgess, int *d_ArrayOfWeights, int *d_Distance, int *d_Parents) {
    unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);
    if (index < N) {
        for (int j = d_ArrayOfIndex[index]; j < d_ArrayOfIndex[index+1]; ++j) {
            int u = d_ArrayOfVectices[index];
            int w = d_ArrayOfWeights[j];
            int dis_u = d_Distance[index];
            int dis_v = d_Distance[d_ArrayOfEdgess[j]];
            if (dis_v == dis_u + w) {
                atomicMin(&d_Parents[d_ArrayOfEdgess[j]], u);
            }
        }
    }
}

// Update d_Parents when Bellman-Ford algorithm is completed (for version 2 and 3)
__global__ void updatePredecessorVersion3(int N, int *d_ArrayOfVectices, int *d_ArrayOfIndex, int *d_ArrayOfEdgess, int *d_ArrayOfWeights, int *d_Distance, int *d_Parents) {
    unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    unsigned int stride = blockDim.x * gridDim.x;

    for (int index = tid; index < N; index += stride) {
        for (int j = d_ArrayOfIndex[index]; j < d_ArrayOfIndex[index+1]; ++j) {
            int u = d_ArrayOfVectices[index];
            int w = d_ArrayOfWeights[j];
            int dis_u = d_Distance[index];
            int dis_v = d_Distance[d_ArrayOfEdgess[j]];
            if (dis_v == dis_u + w) {
                atomicMin(&d_Parents[d_ArrayOfEdgess[j]], u);
            }
        }
    }
}