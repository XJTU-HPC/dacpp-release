#include <iostream>
#include <vector>

using namespace std;

// Matrix multiplication function
void matrixMultiply(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C) {
    int m = A.size();     // Number of rows in A
    int n = A[0].size();  // Number of columns in A / rows in B
    int p = B[0].size();  // Number of columns in B

    // Perform matrix multiplication
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            C[i][j] = 0;  // Initialize C[i][j]
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    // Initialize two matrices A and B
    vector<vector<int>> A = {{1, 2}, {3, 4}};
    vector<vector<int>> B = {{5, 6}, {7, 8}};
    
    // Result matrix C
    vector<vector<int>> C(2, vector<int>(2, 0)); // Initialize C as a 2x2 matrix with values 0

    // Call the matrix multiplication function
    matrixMultiply(A, B, C);

    // Print the result matrix C
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            cout << C[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}
