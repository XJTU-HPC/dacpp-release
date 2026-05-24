#include <iostream>
#include <vector>
#include <cmath>
using namespace std;
int main() {
    // Define matrix size
    const int N = 100; // Modify N to change the matrix size
    const int max_iter = 10;
    const float tolerance = 1e-6;

    // Initialize coefficient matrix A and vector b
    std::vector<float> A(N * N, 0.0f);
    std::vector<float> b(N, 0.0f);
    std::vector<float> x(N, 0.0f);     // Initial solution
    std::vector<float> x_new(N, 0.0f); // Updated solution

    // Auto-initialize A and b
    for (int i = 0; i < N; ++i) {
        A[i * N + i] = 4.0f; // Diagonal elements, ensure diagonal dominance

        if (i > 0) {
            A[i * N + i - 1] = -1.0f; // Lower triangular elements
        }
        if (i < N - 1) {
            A[i * N + i + 1] = -1.0f; // Upper triangular elements
        }

        b[i] = 1.0f; // Initialize vector b, modify as needed
    }

    bool converged = false;
    int iter = 0;

    while (!converged && iter < max_iter) {
        // Compute new solution vector x_new
        for (int i = 0; i < N; ++i) {
            float sigma = 0.0f;
            // Compute current x_new[i]
            for (int j = 0; j < N; ++j) {
                if (j != i) {
                    sigma += A[i * N + j] * x[j];
                }
            }
            x_new[i] = (b[i] - sigma) / A[i * N + i];
        }

        // Check convergence
        float max_error = 0.0f;
        for (int i = 0; i < N; ++i) {
            max_error = std::max(max_error, std::fabs(x_new[i] - x[i]));
        }

        if (max_error < tolerance) {
            converged = true;
        }

        // Update x
        for (int i = 0; i < N; ++i) {
            x[i] = x_new[i];
        }

        ++iter;
    }

    // Output results
    std::cout << "Iterations: " << iter << std::endl;
    std::cout << "Solution vector x:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
