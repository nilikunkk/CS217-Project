#include "main.h"

#include "kernel.cuh"
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

void printCudaDevice(){
    int device = 0;
    cudaSetDevice(device);
    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, device) == 0)
    {
        printf("****** Device information %d ***********\n", device);
        printf("%s; global memory: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem,
               (int)devProps.major, (int)devProps.minor,
               (int)devProps.clockRate);
        printf("Multiprocessors number: %d\n", devProps.multiProcessorCount);
        printf("Dimension of grid MAX size: %d\n", devProps.maxGridSize);
        printf("Dimension of block MAX size: %d\n", devProps.maxThreadsDim);
        printf("Threads per block MAX number: %d\n", devProps.maxThreadsPerBlock);
        printf("Threads per multiprocessor MAX resident: %d\n", devProps.maxThreadsPerMultiProcessor);
        printf("Available shared memory per block(bytes) : %zu \n", devProps.sharedMemPerBlock );
        printf("Available shared memory per multiprocessor(bytes) : %zu \n", devProps.sharedMemPerMultiprocessor );
        printf("Warp size: %d \n", devProps.warpSize );
        printf("***************************************\n");
    }
}

int runBellmanParallelVersion1(const char *file, int blockSize, int debug) {

    std::string inputFile=file;
    int BLOCK_SIZE = blockSize;
    int DEBUG = debug;
    int MAX_VAL = std::numeric_limits<int>::max();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cout << "BellmanFord_version1 start on GPU" << endl;
    cudaEventRecord(start, 0);

    std::vector<int> V, I, E, W;
    //Load data from files
    loadVector((inputFile + "_V.csv").c_str(), V);
    loadVector((inputFile + "_I.csv").c_str(), I);
    loadVector((inputFile + "_E.csv").c_str(), E);
    loadVector((inputFile + "_W.csv").c_str(), W);

    if(DEBUG){
        cout << "V = "; printVector(V); cout << endl;
        cout << "I = "; printVector(I); cout << endl;
        cout << "E = "; printVector(E); cout << endl;
        cout << "W = "; printVector(W); cout << endl;
    }

    int N = I.size();
    int BLOCKS = 1;
    BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printCudaDevice();
    cout << "Blocks: " << BLOCKS << " BlockSize: " << BLOCK_SIZE << endl;

    int *d_ArrayOfVectices; //ArrayOfVectices
    int *d_ArrayOfIndex; //ArrayOfIndex
    int *d_ArrayOfEdgess; //ArrayOfEdgess
    int *d_ArrayOfWeights; //ArrayOfWeights
    int *d_Distance; //Distance
    int *d_DistanceTrack; //DistanceTrack
    int *d_Parents;  //Parents



    //allocate memory
    cudaMalloc((void**) &d_ArrayOfVectices, V.size() *sizeof(int));
    cudaMalloc((void**) &d_ArrayOfIndex, I.size() *sizeof(int));
    cudaMalloc((void**) &d_ArrayOfEdgess, E.size() *sizeof(int));
    cudaMalloc((void**) &d_ArrayOfWeights, W.size() *sizeof(int));

    cudaMalloc((void**) &d_Distance, V.size() *sizeof(int));
    cudaMalloc((void**) &d_DistanceTrack, V.size() *sizeof(int));
    cudaMalloc((void**) &d_Parents, V.size() *sizeof(int));

    //copy to device memory
    cudaMemcpy(d_ArrayOfVectices, V.data(), V.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ArrayOfIndex, I.data(), I.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ArrayOfEdgess, E.data(), E.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ArrayOfWeights, W.data(), W.size() *sizeof(int), cudaMemcpyHostToDevice);

    int INIT_BLOCKS = (V.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializeArrayVersion1 <<<INIT_BLOCKS, BLOCK_SIZE>>>(V.size(), d_Distance, MAX_VAL, true, 0, 0);
    initializeArrayVersion1 <<<INIT_BLOCKS, BLOCK_SIZE>>>(V.size(), d_Parents, MAX_VAL, true, 0, 0);
    initializeArrayVersion1 <<<INIT_BLOCKS, BLOCK_SIZE>>>(V.size(), d_DistanceTrack, MAX_VAL, true, 0, 0);


    INIT_BLOCKS = (E.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    updateEdgesIndexVersion1<<<INIT_BLOCKS, BLOCK_SIZE>>>(E.size(), d_ArrayOfVectices, d_ArrayOfEdgess, 0, V.size()-1);

    // BellmanFord_V1
    for (int round = 1; round < V.size(); round++) {
        if(DEBUG){
            cout<< "***** round = " << round << " ******* " << endl;
        }
        edgeRelaxationVersion1<<<BLOCKS, BLOCK_SIZE>>>(N, MAX_VAL, d_ArrayOfVectices, d_ArrayOfIndex, d_ArrayOfEdgess, d_ArrayOfWeights, d_Distance, d_DistanceTrack);
        updateDistanceVersion1<<<BLOCKS, BLOCK_SIZE>>>(V.size(), d_ArrayOfVectices, d_ArrayOfIndex, d_ArrayOfEdgess, d_ArrayOfWeights, d_Distance, d_DistanceTrack);
    }
    updatePredecessorVersion1<<<BLOCKS, BLOCK_SIZE>>> (V.size(), d_ArrayOfVectices, d_ArrayOfIndex, d_ArrayOfEdgess, d_ArrayOfWeights, d_Distance, d_Parents);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cout << "BellmanFord_Version1 finish on GPU" << endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    int *out_path = new int[V.size()];
    int *out_pred = new int[V.size()];

    cudaMemcpy(out_path, d_Distance, V.size()*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_pred, d_Parents, V.size()*sizeof(int), cudaMemcpyDeviceToHost);

    if(DEBUG) {
        cout << "Shortest Path : " << endl;
        for (int i = 0; i < V.size(); i++) {
            cout << "from " << V[0] << " to " << V[i] << " = " << out_path[i] << " predecessor = " << out_pred[i]
                 << std::endl;
        }
    }

    std::string delimiter = "/";
    size_t pos = 0;
    std::string token;
    while ((pos = inputFile.find(delimiter)) != std::string::npos) {
        token = inputFile.substr(0, pos);
        inputFile.erase(0, pos + delimiter.length());
    }
    storeResult(("../output/" + inputFile + "_ShortestPath_cuda_v1.csv").c_str(),V, out_path, out_pred);
    cout << "Results written to " << ("../output/" + inputFile + "_ShortestPath_cuda_v1.csv").c_str() << endl;
    cout << "** average time elapsed: " << elapsedTime << " million seconds** " << endl;

    free(out_pred);
    free(out_path);
    cudaFree(d_ArrayOfVectices);
    cudaFree(d_ArrayOfIndex);
    cudaFree(d_ArrayOfEdgess);
    cudaFree(d_ArrayOfWeights);
    cudaFree(d_Distance);
    cudaFree(d_Parents);
    cudaFree(d_DistanceTrack);
    return 0;
}

int runBellmanParallelVersion2(const char *file, int blocks, int blockSize, int debug){

    std::string inputFile=file;
    int BLOCKS = blocks;
    int BLOCK_SIZE = blockSize;
    int DEBUG = debug;
    int MAX_VAL = std::numeric_limits<int>::max();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cout << "BellmanFord_Version2 start on GPU" << endl;
    cudaEventRecord(start, 0);

    std::vector<int> V, I, E, W;
    //Load data from files
    loadVector((inputFile + "_V.csv").c_str(), V);
    loadVector((inputFile + "_I.csv").c_str(), I);
    loadVector((inputFile + "_E.csv").c_str(), E);
    loadVector((inputFile + "_W.csv").c_str(), W);

    if(DEBUG){
        cout << "V = "; printVector(V); cout << endl;
        cout << "I = "; printVector(I); cout << endl;
        cout << "E = "; printVector(E); cout << endl;
        cout << "W = "; printVector(W); cout << endl;
    }

    int N = I.size();
    printCudaDevice();
    cout << "Blocks: " << BLOCKS << " Block Size: " << BLOCK_SIZE << endl;

    int * d_ArrayOfVectices;
    int * d_ArrayOfIndex;
    int *d_ArrayOfEdgess;
    int *d_ArrayOfWeights;
    int *d_Distance; 
    int *d_DistanceTrack; 
    int *d_Parents; 

    //allocate memory
    cudaMalloc((void**) &d_ArrayOfVectices, V.size() *sizeof(int));
    cudaMalloc((void**) &d_ArrayOfIndex, I.size() *sizeof(int));
    cudaMalloc((void**) &d_ArrayOfEdgess, E.size() *sizeof(int));
    cudaMalloc((void**) &d_ArrayOfWeights, W.size() *sizeof(int));

    cudaMalloc((void**) &d_Distance, V.size() *sizeof(int));
    cudaMalloc((void**) &d_DistanceTrack, V.size() *sizeof(int));
    cudaMalloc((void**) &d_Parents, V.size() *sizeof(int));

    //copy to device memory
    cudaMemcpy(d_ArrayOfVectices, V.data(), V.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ArrayOfIndex, I.data(), I.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ArrayOfEdgess, E.data(), E.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ArrayOfWeights, W.data(), W.size() *sizeof(int), cudaMemcpyHostToDevice);

    initializeArrayVersion2 <<<BLOCKS, BLOCK_SIZE>>>(V.size(), d_Distance, MAX_VAL, true, 0, 0);
    initializeArrayVersion2 <<<BLOCKS, BLOCK_SIZE>>>(V.size(), d_Parents, MAX_VAL, true, 0, 0);
    initializeArrayVersion2 <<<BLOCKS, BLOCK_SIZE>>>(V.size(), d_DistanceTrack, MAX_VAL, true, 0, 0);

    updateEdgesIndexVersion2<<<BLOCKS, BLOCK_SIZE>>>(E.size(), d_ArrayOfVectices, d_ArrayOfEdgess, 0, V.size()-1);

    // Bellmanford_V2
    for (int round = 1; round < V.size(); round++) {
        if(DEBUG){
            cout<< "***** round = " << round << " ******* " << endl;
        }
        edgeRelaxationVersion2<<<BLOCKS, BLOCK_SIZE>>>(N, MAX_VAL, d_ArrayOfVectices, d_ArrayOfIndex, d_ArrayOfEdgess, d_ArrayOfWeights, d_Distance, d_DistanceTrack);
        updateDistanceVersion2<<<BLOCKS, BLOCK_SIZE>>>(V.size(), d_ArrayOfVectices, d_ArrayOfIndex, d_ArrayOfEdgess, d_ArrayOfWeights, d_Distance, d_DistanceTrack);
    }
    updatePredecessorVersion3<<<BLOCKS, BLOCK_SIZE>>> (V.size(), d_ArrayOfVectices, d_ArrayOfIndex, d_ArrayOfEdgess, d_ArrayOfWeights, d_Distance, d_Parents);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cout << "BellmanFord_Version2 finish on GPU" << endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    int *out_path = new int[V.size()];
    int *out_pred = new int[V.size()];

    cudaMemcpy(out_path, d_Distance, V.size()*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_pred, d_Parents, V.size()*sizeof(int), cudaMemcpyDeviceToHost);

    if(DEBUG) {
        cout << "Shortest Path : " << endl;
        for (int i = 0; i < V.size(); i++) {
            cout << "from " << V[0] << " to " << V[i] << " = " << out_path[i] << " predecessor = " << out_pred[i]
                 << std::endl;
        }
    }

    std::string delimiter = "/";
    size_t pos = 0;
    std::string token;
    while ((pos = inputFile.find(delimiter)) != std::string::npos) {
        token = inputFile.substr(0, pos);
        inputFile.erase(0, pos + delimiter.length());
    }
    storeResult(("../output/" + inputFile + "_ShortestPath_cuda_v2.csv").c_str(),V, out_path, out_pred);
    cout << "Results written to " << ("../output/" + inputFile + "_ShortestPath_cuda_v2.csv").c_str() << endl;
    cout << "** average time elapsed: " << elapsedTime << " million seconds** " << endl;

    free(out_pred);
    free(out_path);
    cudaFree(d_ArrayOfVectices);
    cudaFree(d_ArrayOfIndex);
    cudaFree(d_ArrayOfEdgess);
    cudaFree(d_ArrayOfWeights);
    cudaFree(d_Distance);
    cudaFree(d_Parents);
    cudaFree(d_DistanceTrack);
    return 0;
}


int runBellmanParallelVersion3(const char *file, int blocks, int blockSize, int debug){

    std::string inputFile=file;
    int BLOCKS = blocks;
    int BLOCK_SIZE = blockSize;
    int DEBUG = debug;
    int MAX_VAL = std::numeric_limits<int>::max();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cout << "BellmanFord_Version3 start on GPU" << endl;
    cudaEventRecord(start, 0);

    std::vector<int> V, I, E, W;
    //Load data from files
    loadVector((inputFile + "_V.csv").c_str(), V);
    loadVector((inputFile + "_I.csv").c_str(), I);
    loadVector((inputFile + "_E.csv").c_str(), E);
    loadVector((inputFile + "_W.csv").c_str(), W);

    if(DEBUG){
        cout << "V = "; printVector(V); cout << endl;
        cout << "I = "; printVector(I); cout << endl;
        cout << "E = "; printVector(E); cout << endl;
        cout << "W = "; printVector(W); cout << endl;
    }

    int N = I.size();
    printCudaDevice();
    cout << "Blocks : " << BLOCKS << " Block size: " << BLOCK_SIZE << endl;

    int * d_ArrayOfVectices;
    int * d_ArrayOfIndex;
    int *d_ArrayOfEdgess;
    int *d_ArrayOfWeights;
    int *d_Distance; 
    int *d_DistanceTrack; 
    int *d_Parents; 
    bool *d_Flag;

    //allocate memory
    cudaMalloc((void**) &d_ArrayOfVectices, V.size() *sizeof(int));
    cudaMalloc((void**) &d_ArrayOfIndex, I.size() *sizeof(int));
    cudaMalloc((void**) &d_ArrayOfEdgess, E.size() *sizeof(int));
    cudaMalloc((void**) &d_ArrayOfWeights, W.size() *sizeof(int));

    cudaMalloc((void**) &d_Distance, V.size() *sizeof(int));
    cudaMalloc((void**) &d_DistanceTrack, V.size() *sizeof(int));
    cudaMalloc((void**) &d_Parents, V.size() *sizeof(int));
    cudaMalloc((void**) &d_Flag, V.size() *sizeof(bool));

    //copy to device memory
    cudaMemcpy(d_ArrayOfVectices, V.data(), V.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ArrayOfIndex, I.data(), I.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ArrayOfEdgess, E.data(), E.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ArrayOfWeights, W.data(), W.size() *sizeof(int), cudaMemcpyHostToDevice);

    initializeArrayVersion2 <<<BLOCKS, BLOCK_SIZE>>>(V.size(), d_Distance, MAX_VAL, true, 0, 0);
    initializeArrayVersion2 <<<BLOCKS, BLOCK_SIZE>>>(V.size(), d_Parents, MAX_VAL, true, 0, 0);
    initializeArrayVersion2 <<<BLOCKS, BLOCK_SIZE>>>(V.size(), d_DistanceTrack, MAX_VAL, true, 0, 0);
    initializeBooleanArrayWithGridStride<<<BLOCKS, BLOCK_SIZE>>>(V.size(), d_Flag, false, true, 0, true); // set all elements to false except source which is V[0]


    updateEdgesIndexVersion2<<<BLOCKS, BLOCK_SIZE>>>(E.size(), d_ArrayOfVectices, d_ArrayOfEdgess, 0, V.size()-1);

    // Bellmanford_V3
    for (int round = 1; round < V.size(); round++) {
        if(DEBUG){
            cout<< "***** round = " << round << " ******* " << endl;
        }
        edgeRelaxationVersion3<<<BLOCKS, BLOCK_SIZE>>>(N, MAX_VAL, d_ArrayOfVectices, d_ArrayOfIndex, d_ArrayOfEdgess, d_ArrayOfWeights, d_Distance, d_DistanceTrack, d_Flag);
        updateDistanceVersion3<<<BLOCKS, BLOCK_SIZE>>>(V.size(), d_ArrayOfVectices, d_ArrayOfIndex, d_ArrayOfEdgess, d_ArrayOfWeights, d_Distance, d_DistanceTrack, d_Flag);
    }
    updatePredecessorVersion3<<<BLOCKS, BLOCK_SIZE>>> (V.size(), d_ArrayOfVectices, d_ArrayOfIndex, d_ArrayOfEdgess, d_ArrayOfWeights, d_Distance, d_Parents);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cout << "BellmanFord_Version3 finish on GPU" << endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    int *out_path = new int[V.size()];
    int *out_pred = new int[V.size()];

    cudaMemcpy(out_path, d_Distance, V.size()*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_pred, d_Parents, V.size()*sizeof(int), cudaMemcpyDeviceToHost);

    if(DEBUG) {
        cout << "Shortest Path: " << endl;
        for (int i = 0; i < V.size(); i++) {
            cout << "from " << V[0] << " to " << V[i] << " = " << out_path[i] << " predecessor = " << out_pred[i]
                 << std::endl;
        }
    }

    std::string delimiter = "/";
    size_t pos = 0;
    std::string token;
    while ((pos = inputFile.find(delimiter)) != std::string::npos) {
        token = inputFile.substr(0, pos);
        inputFile.erase(0, pos + delimiter.length());
    }
    storeResult(("../output/" + inputFile + "_ShortestPath_cuda_v3.csv").c_str(),V, out_path, out_pred);
    cout << "Results written to " << ("../output/" + inputFile + "_ShortestPath_cuda_v3.csv").c_str() << endl;
    cout << "** average time elapsed : " << elapsedTime << " million seconds** " << endl;

    free(out_pred);
    free(out_path);
    cudaFree(d_ArrayOfVectices);
    cudaFree(d_ArrayOfIndex);
    cudaFree(d_ArrayOfEdgess);
    cudaFree(d_ArrayOfWeights);
    cudaFree(d_Distance);
    cudaFree(d_Parents);
    cudaFree(d_DistanceTrack);
    return 0;
}