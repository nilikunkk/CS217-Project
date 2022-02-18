#include "main.h"

int main(int argc, char **argv) {
    if (argc < 2 ){
        cout << "Usage: ./bellman MODE(M) FILE(F) BLOCK_SIZE(B) DEBUG(D)" << endl;
        cout << "M: sequential/parallel \n"
                "F: input file \n"
                "B: block numbers each grid\n"
                "B: threads numbers each block \n"
                "D: 1or0 to enable/unable extended debug messages on console\n"
                "CSV files are expected based on FILE\n"
                "    FILE_V.csv\n"
                "    FILE_I.csv\n"
                "    FILE_E.csv\n"
                "    FILE_W.csv"
                << endl;
        return -1;
    }
    std::string mode = argv[1];
    std::string file;
    if(argv[2] != NULL){
        file = argv[2];
        //Check if all CSR files are present
        if(!isValidFile(file + "_V.csv") ||
           !isValidFile(file + "_I.csv") ||
           !isValidFile(file + "_E.csv") ||
           !isValidFile(file + "_W.csv")){
            cout << "CSR files missing" << endl;
            return -1;
        }
    }

    int BLOCK = argc > 3 ? atoi(argv[3]) : 4;
    int BLOCK_SIZE = argc > 4 ? atoi(argv[4]) : 512;
    int DEBUG_MODE = argc > 5 ? atoi(argv[5]) : 0;

    if(mode == "sequential") {
        auto start = high_resolution_clock::now();
        runBellmanSequential(file, DEBUG_MODE);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << "Elapsed time: " << duration.count() << " million seconds " << endl;
    }
    if(mode == "CUDA-V1"){
        runBellmanParallelVersion1(file.c_str(), BLOCK_SIZE, DEBUG_MODE);
    }
    if(mode == "CUDA-V2"){
        runBellmanParallelVersion2(file.c_str(), BLOCK, BLOCK_SIZE, DEBUG_MODE);
    }
    if(mode == "CUDA-V3") {
        runBellmanParallelVersion3(file.c_str(), BLOCK, BLOCK_SIZE, DEBUG_MODE);

    }

}