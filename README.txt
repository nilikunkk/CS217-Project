Compiling and running the project
1. mkdir build
2. cd build
3. cmake ../.
4. cmake --build .
5. cd Debug
6. ./bellman CUDA-V1 ../../input/USA-road-d.NY.gr 196 1024 0 ('196' is the parameter for the number of blocks and '1024' is the parameter for the size of block)
7. ./bellman CUDA-V2 ../../input/USA-road-d.NY.gr 196 1024 0
8. ./bellman CUDA-V3 ../../input/USA-road-d.NY.gr 196 1024 0

Input Data
Data from http://users.diag.uniroma1.it/challenge9/download.shtml, we preprocess these dataset, get four kind of dataset:
USA-road-d.NY.gr_V.csv - Contains V array
USA-road-d.NY.gr_I.csv - Contains I array
USA-road-d.NY.gr_E.csv - Contains E array
USA-road-d.NY.gr_W.csv - Contains W array
