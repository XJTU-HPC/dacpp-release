#include <iostream>
#include <vector>
#include "ReconTensor.h"
#include <string>
#include <filesystem>




namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;

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
    for (int i = 0; i < 5; i++) {
        sum += vecA[i] * vecB[i];
    }
    dotProduct[0] = sum;
}

int main() {
    std::string dir = std::filesystem::path(__FILE__).parent_path().string();

    auto objA = dacpp::read_matrix_mtx_col<int>(dir + "/dataA.mtx");
    auto objB = dacpp::read_matrix_mtx_col<int>(dir + "/dataB.mtx");

    dacpp::Matrix<int> matA(objA.shape, objA.data);
    dacpp::Matrix<int> matB(objB.shape, objB.data);

    std::vector<int> dataC(objA.shape[0] * objB.shape[1], 0);
    dacpp::Matrix<int> matC({objA.shape[0], objB.shape[1]}, dataC);

    matrixMultiply_shell(matA, matB, matC) <-> matrixMultiply_calc;

    matC.print();

    return 0;
}
