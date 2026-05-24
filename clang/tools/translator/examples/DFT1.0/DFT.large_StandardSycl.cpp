// dft_sycl.cpp
#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <cmath>

using namespace sycl;
using namespace std;

int main() {
    constexpr int N = 8192;

    // --- Prepare input: using a real sequence as example (imag = 0), modify as needed ---
    std::vector<double> in_real(N), in_imag(N);
    for (int n = 0; n < N; ++n) {
        in_real[n] = static_cast<double>(n); // Example: 0,1,2,...,N-1
        in_imag[n] = 0.0;
    }

    // Output buffers (real/imaginary parts)
    std::vector<double> out_real(N, 0.0), out_imag(N, 0.0);

    // Create SYCL queue (select default device)
    queue q;

    {
        // Create buffers (RAII)
        buffer<double, 1> buf_in_real(in_real.data(), range<1>(N));
        buffer<double, 1> buf_in_imag(in_imag.data(), range<1>(N));
        buffer<double, 1> buf_out_real(out_real.data(), range<1>(N));
        buffer<double, 1> buf_out_imag(out_imag.data(), range<1>(N));

        // Submit task: compute X[k] in parallel for each k
        q.submit([&](handler &h) {
            accessor a_in_r(buf_in_real, h, read_only);
            accessor a_in_i(buf_in_imag, h, read_only);
            accessor a_out_r(buf_out_real, h, write_only, no_init);
            accessor a_out_i(buf_out_imag, h, write_only, no_init);

            h.parallel_for<class dft_kernel>(range<1>(N), [=](id<1> idk) {
                int k = idk[0];
                double sum_r = 0.0;
                double sum_i = 0.0;

                // Inner n loop executes sequentially on device
                for (int n = 0; n < N; ++n) {
                    // Angle: angle = -2 * pi * k * n / N
                    double angle = -2.0 * M_PI * k * n / static_cast<double>(N);
                    double c = sycl::cos(angle);
                    double s = sycl::sin(angle);

                    double a = a_in_r[n];
                    double b = a_in_i[n];

                    // (a + i b) * (c + i s) = (a*c - b*s) + i(a*s + b*c)
                    sum_r += a * c - b * s;
                    sum_i += a * s + b * c;
                }

                a_out_r[k] = sum_r;
                a_out_i[k] = sum_i;
            });
        });

        // Wait for queue to complete, buffer destructor automatically copies data back to host
        q.wait();
    } // buffers go out of scope and sync back to out_real/out_imag

    // Formatted output: each element as (real, imag)
    std::cout << "{";
    for (int k = 0; k < N; ++k) {
        std::cout << "(" << out_real[k] << "," << out_imag[k] << ")";
        if (k < N - 1) std::cout << ", ";
    }
    std::cout << "}" << std::endl;

    return 0;
}
