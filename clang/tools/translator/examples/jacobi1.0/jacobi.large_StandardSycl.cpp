#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>


using namespace sycl;

int main() {
    auto start_time = std::chrono::high_resolution_clock::now(); // Start time measurement
    const int N = 1000; // Matrix size
    const int max_iter = 10000;
    const float tolerance = 1e-6;

    std::vector<float> A(N * N, 0.0f);
    std::vector<float> b(N, 0.0f);
    std::vector<float> x(N, 0.0f);     // Initial solution
    std::vector<float> x_new(N, 0.0f); // Updated solution

    // Initialize A and b
    for (int i = 0; i < N; ++i) {
        A[i * N + i] = 4.0f; // Diagonal elements
        if (i > 0) A[i * N + i - 1] = -1.0f; // Lower triangular
        if (i < N - 1) A[i * N + i + 1] = -1.0f; // Upper triangular
        b[i] = 1.0f; // Vector b
    }

    queue q;

    buffer<float, 1> A_buf(A.data(), range<1>(N * N));
    buffer<float, 1> b_buf(b.data(), range<1>(N));
    buffer<float, 1> x_buf(x.data(), range<1>(N));
    buffer<float, 1> x_new_buf(x_new.data(), range<1>(N));

    bool converged = false;
    int iter = 0;

    while (!converged && iter < max_iter) {
        //std::cout << "Iterations: " << iter << std::endl;

        // Submit computation task
        q.submit([&](handler& h) {
            auto A_acc = A_buf.get_access<access::mode::read>(h);
            auto b_acc = b_buf.get_access<access::mode::read>(h);
            auto x_acc = x_buf.get_access<access::mode::read>(h);
            auto x_new_acc = x_new_buf.get_access<access::mode::discard_write>(h);

            h.parallel_for(range<1>(N), [=](id<1> i) {
                float sigma = 0.0f;
                for (int j = 0; j < N; ++j) {
                    if (j != i[0]) {
                        sigma += A_acc[i[0] * N + j] * x_acc[j];
                    }
                }
                x_new_acc[i] = (b_acc[i] - sigma) / A_acc[i[0] * N + i[0]];
            });
        });

        // Wait for computation to complete
        q.wait();

        // Check convergence
        auto x_acc_host = x_buf.get_access<access::mode::read>();
        auto x_new_acc_host = x_new_buf.get_access<access::mode::read>();

        float max_error = 0.0f;
        for (int i = 0; i < N; ++i) {
            max_error = std::max(max_error, std::fabs(x_new_acc_host[i] - x_acc_host[i]));
        }
        //std::cout << "Max error: " << max_error << std::endl;

        if (max_error < tolerance) {
            converged = true;
        }

        // Update x on host side
        for (int i = 0; i < N; ++i) {
            x[i] = x_new[i];
        }

        ++iter;
    }

    // Get final results
    auto x_result = x_new_buf.get_access<access::mode::read>();
    //std::cout << "Iterations: " << iter << std::endl;
    //std::cout << "Solution vector x:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << x_result[i] << " ";
    }
    std::cout << std::endl;
    auto end_time = std::chrono::high_resolution_clock::now(); // End time measurement
    std::chrono::duration<double> duration = end_time - start_time; // Calculate duration
    //std::cout << "Total execution time: " << duration.count() << " seconds" << std::endl;
    return 0;
}
