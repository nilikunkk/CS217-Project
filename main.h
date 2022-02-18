#ifndef GPU_GRAPH_ALGORITHMS_MAIN_H
#define GPU_GRAPH_ALGORITHMS_MAIN_H

#include <algorithm>
#include <cassert>
#include <vector>
#include <limits>
#include <stdio.h>
#include <string>
#include <string.h>
#include <ctime>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include "utilities.h"

using namespace std;
using namespace std::chrono;
using std::cout;
using std::endl;

void runBellmanSequential(std::string file, int debug);
int runBellmanParallelVersion1(const char *file, int blockSize, int debug);
int runBellmanParallelVersion2(const char *file, int blocks, int blockSize, int debug);
int runBellmanParallelVersion3(const char *file, int blocks, int blockSize, int debug);

#endif //GPU_GRAPH_ALGORITHMS_MAIN_H
