#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;
const int N = 8192;
shell dacpp::list matrixMultiply_shell( const dacpp::Matrix<int>& matA,
                                        const dacpp::Matrix<int>& matB,
                                        dacpp::Matrix<int>& matC) {
    dacpp::index idx1, idx2;
    dacpp::list dataList{matA[idx1][{}], matB[{}][idx2], matC[idx1][idx2]};
    return dataList;
}

calc void matrixMultiply_calc(dacpp::Vector<int>& vecA,
                dacpp::Vector<int>& vecB,
                int* dotProduct) {
    int sum = 0;
    for (int i = 0; i < N; i++) {
        sum += vecA[i] * vecB[i];
    }
    dotProduct[0] = sum;
}

int main() {
    // Initialize two matrices A and B
        // Generate all-ones matrix A (N x N)
    std::vector<int> dataA(N * N, 1);
    dacpp::Matrix<int> matA({N, N}, dataA);
    
    // Generate all-ones matrix B (N x N)
    std::vector<int> dataB(N * N, 1);
    dacpp::Matrix<int> matB({N, N}, dataB);
    
    // Generate result matrix C (N x N), initially all zeros
    std::vector<int> dataC(N * N, 0);
    dacpp::Matrix<int> matC({N, N}, dataC);

    matrixMultiply_shell(matA, matB, matC) <-> matrixMultiply_calc;

    matC.print();

    return 0;
}
