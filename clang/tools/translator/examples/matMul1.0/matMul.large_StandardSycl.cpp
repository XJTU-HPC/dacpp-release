#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace sycl;
const size_t M = 8192;
const size_t N = 8192;
const size_t K = 8192;
const size_t TILE_SIZE = 16; // Local work group tile size (aligned with the first code segment)

void matrix_multiply_sycl(
    const std::vector<int>& A,
    const std::vector<int>& B,
    std::vector<int>& C
) {
    // 1. Create SYCL queue and print device info
    sycl::queue q(sycl::default_selector_v);
    auto device = q.get_device();
    //std::cout << "Using compute device: " << device.get_info<info::device::name>() << std::endl;

    // Get the maximum local work item size supported by the device (to avoid exceeding device limits)
    auto max_local_sizes = device.get_info<info::device::max_work_item_sizes<2>>();
    size_t max_local_x = max_local_sizes[0];
    size_t max_local_y = max_local_sizes[1];
    size_t local_x = std::min(TILE_SIZE, max_local_x);
    size_t local_y = std::min(TILE_SIZE, max_local_y);
    //std::cout << "Using local work group size: " << local_x << "x" << local_y << std::endl;

    // 2. Calculate aligned global work item size (must be an integer multiple of local size, consistent with the first code segment logic)
    size_t global_x = ((M + local_x - 1) / local_x) * local_x;
    size_t global_y = ((N + local_y - 1) / local_y) * local_y;
    //std::cout << "Aligned global work item size: " << global_x << "x" << global_y << std::endl;

    // 3. Create 2D SYCL buffer (bound to host-side vector data)
    sycl::buffer<int, 2> buf_A(A.data(), sycl::range<2>(M, K));
    sycl::buffer<int, 2> buf_B(B.data(), sycl::range<2>(K, N));
    sycl::buffer<int, 2> buf_C(C.data(), sycl::range<2>(M, N));

    // 4. Submit SYCL command group (using nd_range for global/local optimization)
    q.submit([&](sycl::handler& h) {
        // 4.1 Create accessors (device-side interface for accessing buffers)
        auto acc_A = buf_A.get_access<sycl::access::mode::read>(h);
        auto acc_B = buf_B.get_access<sycl::access::mode::read>(h);
        auto acc_C = buf_C.get_access<sycl::access::mode::write>(h);

        // 4.2 Define nd_range (global range + local work group range)
        sycl::range<2> global_range(global_x, global_y);
        sycl::range<2> local_range(local_x, local_y);
        sycl::nd_range<2> nd_range(global_range, local_range);

        // 4.3 Parallel kernel function (using nd_item<2>, supporting global/local indexing)
        h.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            // Get global index (corresponding to row i, column j of matrix C)
            size_t i = item.get_global_id(0);
            size_t j = item.get_global_id(1);

            if (i < M && j < N) {
                int sum = 0;
                // Compute the dot product sum for row i, column j (preserving original logic, can be further optimized with local memory later)
                for (size_t k = 0; k < K; ++k) {
                    sum += acc_A[i][k] * acc_B[k][j];
                }
                acc_C[i][j] = sum;
            }
        });
    }).wait(); // Wait for kernel execution to complete
}

int main() {
    // double total_program = 0;
    // auto program_start = std::chrono::high_resolution_clock::now();

    // Initialize matrix data (A: M*K, B: K*N, C: M*N)
    std::vector<int> dataA(M * K, 1);  // A is M rows by K columns, all elements are 1
    std::vector<int> dataB(K * N, 1);  // B is K rows by N columns, all elements are 1
    std::vector<int> dataC(M * N, 0);  // C is M rows by N columns, initially all zeros

    matrix_multiply_sycl(dataA, dataB, dataC);

    // Formatted output: {{a, b, c, d}, {e, f, g, h}, ...}
    std::cout << "{";
    for (int i = 0; i < M; i++) {
        std::cout << "{";
        for (int j = 0; j < N; j++) {
            std::cout << dataC[i * N + j];
            if (j < N - 1) std::cout << ", ";
        }
        std::cout << "}";
        if (i < M - 1) std::cout << ", ";
    }
    std::cout << "}" << std::endl;
    // Calculate and print total elapsed time
    // auto program_end = std::chrono::high_resolution_clock::now();
    // total_program += std::chrono::duration<double>(program_end - program_start).count();
    // std::cout << "Total runtime: " << total_program << " seconds" << std::endl;

    return 0;
}