#include <CL/sycl.hpp>
#include <vector>
#include <iostream>

using namespace sycl;

int main() {
    constexpr int B = 2;   // batch size
    constexpr int M = 4;   // number of rows
    constexpr int K = 5;   // reduction dimension
    constexpr int N = 4;   // number of columns

    // =========================
    // Batch 0
    // =========================
    // A[0]: 4 × 5
    // Row-major order
    std::vector<int> dataA{
        // ---- Batch 0 ----
        10, 11, 12, 13, 14,
        11, 12, 13, 14, 15,
        12, 13, 14, 15, 16,
        13, 14, 15, 16, 17,

        // ---- Batch 1 ----
        20, 21, 22, 23, 24,
        21, 22, 23, 24, 25,
        22, 23, 24, 25, 26,
        23, 24, 25, 26, 27
    };

    // B[b]: 5 x 4 matrix
    std::vector<int> dataB{
        // ---- Batch 0 ----
        10, 11, 12, 13,
        11, 12, 13, 14,
        12, 13, 14, 15,
        13, 14, 15, 16,
        14, 15, 16, 17,

        // ---- Batch 1 ----
        20, 21, 22, 23,
        21, 22, 23, 24,
        22, 23, 24, 25,
        23, 24, 25, 26,
        24, 25, 26, 27
    };

    // Result matrix C: [B][M][N]
    std::vector<int> result(B * M * N, 0);

    {
        queue q;

        buffer<int, 1> bufA(dataA.data(), range<1>(B * M * K));
        buffer<int, 1> bufB(dataB.data(), range<1>(B * K * N));
        buffer<int, 1> bufC(result.data(), range<1>(B * M * N));

        q.submit([&](handler& h) {
            accessor a(bufA, h, read_only);
            accessor b(bufB, h, read_only);
            accessor c(bufC, h, write_only, no_init);

            h.parallel_for(range<3>(B, M, N), [=](id<3> idx) {
                int bb = idx[0];  // batch index
                int i  = idx[1];  // row index
                int j  = idx[2];  // column index

                int sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += a[bb * M * K + i * K + k]
                         * b[bb * K * N + k * N + j];
                }

                c[bb * M * N + i * N + j] = sum;
            });
        });

        q.wait();
    }

    // =========================
    // Modified output section: match the {{{...}, {...}}, {{...}, {...}}} format
    // =========================
    std::cout << "{";
    for (int b = 0; b < B; b++) {
        std::cout << "{";
        for (int i = 0; i < M; i++) {
            std::cout << "{";
            for (int j = 0; j < N; j++) {
                std::cout << result[b * M * N + i * N + j];
                if (j < N - 1) std::cout << ", ";
            }
            std::cout << "}";
            if (i < M - 1) std::cout << ", ";
        }
        std::cout << "}";
        if (b < B - 1) std::cout << ", ";
    }
    std::cout << "}" << std::endl;

    return 0;
}