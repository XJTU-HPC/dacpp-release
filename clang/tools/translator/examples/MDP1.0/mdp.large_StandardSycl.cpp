#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// Parameter settings
const double A = 1.0;  // Attraction coefficient
const double D = 0.1;  // Diffusion coefficient
const double dx = 0.1; // Spatial step size
const double dt = 0.01; // Time step size
const int N = 10000;     // Number of spatial grid points
const int T = 1000;    // Number of time steps

// Initialize user preference distribution
void initialize(std::vector<double>& p) {
    for (int i = 0; i < N; ++i) {
        // Assume initial preference follows a Gaussian distribution
        double x = i * dx;
        p[i] = exp(-pow(x - 5.0, 2) / 2.0); // Initial preference distribution centered at x=5
    }
}


// Normalization function
void normalize(std::vector<double>& p) {
    double sum = 0.0;
    for (double value : p) {
        sum += value;
    }
    for (double& value : p) {
        value /= sum; // Normalize
    }
}

int main() {
    std::vector<double> p(N, 0.0); // Store user preference distribution
    std::vector<double> new_p(N, 0.0); // Store distribution at next time step

    // Initialize preference distribution
    initialize(p);

    // Use SYCL for parallel computation
    {
        cl::sycl::queue q;

        for (int t = 0; t < T; ++t) {
            // Update interior points, boundary conditions not considered for now
            cl::sycl::buffer<double, 1> p_buffer(p.data(), cl::sycl::range<1>(N));
            cl::sycl::buffer<double, 1> new_p_buffer(new_p.data(), cl::sycl::range<1>(N));

            q.submit([&](cl::sycl::handler& cgh) {
                auto p_access = p_buffer.get_access<cl::sycl::access::mode::read>(cgh);
                auto new_p_access = new_p_buffer.get_access<cl::sycl::access::mode::write>(cgh);

                cgh.parallel_for<class FokkerPlanck>(cl::sycl::range<1>(N - 2), [=](cl::sycl::item<1> item) {
                    int i = item.get_id(0) + 1; // Because we compute from index 1 to N-2

                    // Diffusion and drift terms
                    double diffusion = D * (p_access[i + 1] - 2 * p_access[i] + p_access[i - 1]) / (dx * dx);
                    double drift = -A * (p_access[i + 1] - p_access[i - 1]) / (2 * dx);
                    new_p_access[i] = p_access[i] + dt * (diffusion + drift);
                });
            });

            // Normalize distribution
            //normalize(p);

            // Update p sequentially, since there are no concurrency conflicts
            for (int i = 1; i < N - 1; ++i) {
                p[i] = new_p[i];  // Update interior points
            }
            // Set boundary conditions
            //p[0] = 0.0;
            //p[N - 1] = 0.0;
        }
        // std::cout << "{";
        // for (int i = 2; i < 2; ++i) {
        //     std::cout << p[i];
        //     //if (i < N - 1) std::cout << ", ";
        // }
        // std::cout << "}" << std::endl;

        std::cout << p[2] << std::endl;
        

            
    }

    return 0;
}
