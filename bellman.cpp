#include "main.h"

void updateEdgeIndex(std::vector<int> &V, std::vector<int> &E, int l, int r){

    for (int index = 0; index < E.size(); index++) {
        l=0; r=V.size()-1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (V[m] < E[index]) {
                l = m + 1;
            }
            if (V[m] == E[index]) {
                E[index] = m;
                break;
            } else {
                r = m - 1;
            }
        }
    }
}
void runBellmanSequential(std::string file, int debug){

    cout << "Running BellmanFord Sequential: " << file << endl;
    std::vector<int> V, I, E, W;

    loadVector((file + "_V.csv").c_str(), V);
    loadVector((file + "_I.csv").c_str(), I);
    loadVector((file + "_E.csv").c_str(), E);
    loadVector((file + "_W.csv").c_str(), W);

    std::vector<int> D(V.size(), std::numeric_limits<int>::max());
    std::vector<int> predecessor(V.size(), -1);

    updateEdgeIndex(V, E, 0, V.size()-1);

    D[0] = 0;
    predecessor[0] = 0;

    //BellmanFord algorithm
    for (int round = 1; round < V.size(); round++) {
        if(debug){
            cout<< "***** round = " << round << " ******* " << endl;
        }
        for (int i = 0; i < I.size()-1 ; i++) {
            for (int j = I[i]; j < I[i + 1]; j++) {
                int u = V[i];
                int v = V[E[j]];
                int w = W[j];
                int du = D[i];
                int dv = D[E[j]];
                if (du + w < dv) {
                    D[E[j]] = du + w;
                    predecessor[E[j]] = u;
                }
            }
        }
    }

    storeResult(("../output/" + makeOutputFileName(file) + "_SP_Sequential.csv").c_str(), V,D.data(),predecessor.data());
    cout << "Results written to " << ("../output/" + makeOutputFileName(file) + "_SP_Sequential.csv").c_str() << endl;

    if (debug) {
        cout << "Shortest Path : " << endl;
        for (int i = 0; i < V.size(); i++) {
            cout << "from " << V[0] << " to " << V[i] << " = " << D[i] << " predecessor = " << predecessor[i] << std::endl;
        }
    }
}