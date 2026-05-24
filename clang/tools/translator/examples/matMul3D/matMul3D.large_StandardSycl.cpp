#include <CL/sycl.hpp>
#include <vector>
#include <iostream>

using namespace sycl;

int main() {
    constexpr int B = 1024;   // batch size
    constexpr int M = 1024;   // number of rows
    constexpr int K = 1024;   // reduction dimension
    constexpr int N = 1024;   // number of columns

    std::vector<int> dataA(B * M * K, 1);
    std::vector<int> dataB(B * K * N, 1);
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