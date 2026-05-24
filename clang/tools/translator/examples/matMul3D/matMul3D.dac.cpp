#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;

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
    for (int i = 0; i < 5; i++) {   // K = 5 (reduction dimension)
        sum += vecA[i] * vecB[i];
    }
    dotProduct[0] = sum;
}
int main() {
    // B = 2, M = 4, K = 5, N = 4
    std::vector<int> dataA{
        // ---- Batch 0 ----
        10,11,12,13,14,
        11,12,13,14,15,
        12,13,14,15,16,
        13,14,15,16,17,

        // ---- Batch 1 ----
        20,21,22,23,24,
        21,22,23,24,25,
        22,23,24,25,26,
        23,24,25,26,27
    };

    std::vector<int> dataB{
        // ---- Batch 0 ----
        10,11,12,13,
        11,12,13,14,
        12,13,14,15,
        13,14,15,16,
        14,15,16,17,

        // ---- Batch 1 ----
        20,21,22,23, 
        21,22,23,24,
        22,23,24,25,
        23,24,25,26,
        24,25,26,27
    };

    std::vector<int> dataC(2 * 4 * 4, 0);

    dacpp::Tensor<int, 3> matA({2, 4, 5}, dataA);
    dacpp::Tensor<int, 3> matB({2, 5, 4}, dataB);
    dacpp::Tensor<int, 3> matC({2, 4, 4}, dataC);

    matrixMultiply3D_shell(matA, matB, matC)
        <-> matrixMultiply3D_calc;

    matC.print();

    return 0;
}