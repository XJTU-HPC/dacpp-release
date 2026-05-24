#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;
const int N = 1024;
// =====================
// Shell: 3D parallel mapping
// =====================
shell dacpp::list matrixMultiply3D_shell(
    const dacpp::Tensor<int, 3>& matA,   // [B][M][K]
    const dacpp::Tensor<int, 3>& matB,   // [B][K][N]
    dacpp::Tensor<int, 3>& matC           // [B][M][N]
) {
    dacpp::index b, i, j;

    dacpp::list dataList{
        matA[b][i][{}],   // Row i of the b-th matrix → Vector<K>
        matB[b][{}][j],   // Column j of the b-th matrix → Vector<K>
        matC[b][i][j]     // C[b][i][j] → scalar
    };

    return dataList;
}
// =====================
// Calc: minimum computation unit
// =====================
calc void matrixMultiply3D_calc(
    dacpp::Vector<int>& vecA,
    dacpp::Vector<int>& vecB,
    int* dotProduct
) {
    int sum = 0;
    for (int i = 0; i < N; i++) {   // K = 5 (reduction dimension)
        sum += vecA[i] * vecB[i];
    }
    dotProduct[0] = sum;
}
int main() {
    // Initialize two matrices A and B
        // Generate all-ones matrix A (N x N)
    std::vector<int> dataA(N * N * N, 1);
    dacpp::Tensor<int, 3> matA({N, N, N}, dataA);
    
    // Generate all-ones matrix B (N x N)
    std::vector<int> dataB(N * N * N, 1);
    dacpp::Tensor<int, 3> matB({N, N, N}, dataB);
    
    // Generate result matrix C (N x N), initialized to all zeros
    std::vector<int> dataC(N * N * N, 0);
    dacpp::Tensor<int, 3> matC({N, N, N}, dataC);

    matrixMultiply3D_shell(matA, matB, matC) <-> matrixMultiply3D_calc;

    matC.print();

    return 0;
}