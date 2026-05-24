#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace sycl;
using namespace std;

// Grid parameters
const int NX = 8192;           // Number of grid points in x direction
const int NY = 8192;           // Number of grid points in y direction
const double Lx = 10.0f;       // Length in x direction
const double Ly = 10.0f;       // Length in y direction
const double alpha = 0.01f;    // Thermal diffusion coefficient
const int TIME_STEPS = 100;  // Number of time steps


int main() {
    // Spatial step size
    double dx = Lx / (NX - 1);
    double dy = Ly / (NY - 1);

    // Stability condition
    double dt_stability = (dx * dx * dy * dy) / (2.0f * alpha * (dx * dx + dy * dy));
    double delta_t = 0.4f * dt_stability; // Choose a more restrictive time step to ensure stability

    //cout << "Grid size: " << NX << "x" << NY << "\n";
    //cout << "dx = " << dx << ", dy = " << dy << ", delta_t = " << delta_t << "\n";

    // Initialize temperature field
    vector<double> u_prev(NX * NY, 0.0f); // Previous step (in the heat equation, only current and previous steps exist)
    vector<double> u_curr(NX * NY, 0.0f); // Current step

    // Initial condition: e.g., a Gaussian heat source at the center
    int cx = NX / 2;
    int cy = NY / 2;
    double sigma = 1.0f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            double x = i * dx;
            double y = j * dy;
            // Gaussian distribution
            u_curr[i * NY + j] = std::exp(-((x - Lx/2.0f)*(x - Lx/2.0f) + (y - Ly/2.0f)*(y - Ly/2.0f)) / (2.0f * sigma * sigma));
        }
    }

    // SYCL queue
    queue q(default_selector_v);
    // cout << "Running on " 
    //      << q.get_device().get_info<info::device::name>() << "\n";

    // Allocate device memory
    double *d_u_prev = malloc_device<double>(NX * NY, q);
    double *d_u_curr = malloc_device<double>(NX * NY, q);
    double *d_u_next = malloc_device<double>(NX * NY, q);

    // Copy initial data to device
    q.memcpy(d_u_prev, u_prev.data(), sizeof(double) * NX * NY).wait();
    q.memcpy(d_u_curr, u_curr.data(), sizeof(double) * NX * NY).wait();

    // Main iteration loop
    for(int t = 0; t < TIME_STEPS; ++t) {
        q.submit([&](handler &h) {
            // Use parallel_for to update interior points
            h.parallel_for<class heat_compute>(
                range<2>{NX - 2, NY - 2},
                [=](id<2> idx) {
                    int i = idx[0] + 1;
                    int j = idx[1] + 1;

                    // Second-order central difference
                    double u_xx = (d_u_curr[(i + 1) * NY + j] - 2.0f * d_u_curr[i * NY + j] + d_u_curr[(i - 1) * NY + j]) / (dx * dx);
                    double u_yy = (d_u_curr[i * NY + (j + 1)] - 2.0f * d_u_curr[i * NY + j] + d_u_curr[i * NY + (j - 1)]) / (dy * dy);

                    // Explicit update formula
                    d_u_next[i * NY + j] = d_u_curr[i * NY + j] + alpha * delta_t * (u_xx + u_yy);
                }
            );
        }).wait();

        // Handle boundary conditions (simple adiabatic boundary, zero derivative)
        // Boundary condition types can be modified as needed

        // Top and bottom boundaries (i = 0 and i = NX-1)
        q.submit([&](handler &h) {
            h.parallel_for<class boundary_top_bottom>(range<1>{NY}, [=](id<1> j) {
                // Top boundary i = 0
                d_u_next[0 * NY + j] = d_u_next[1 * NY + j];
                // Bottom boundary i = NX-1
                d_u_next[(NX - 1) * NY + j] = d_u_next[(NX - 2) * NY + j];
            });
        }).wait();

        // Left and right boundaries (j = 0 and j = NY-1)
        q.submit([&](handler &h) {
            h.parallel_for<class boundary_left_right>(range<1>{NX}, [=](id<1> i) {
                // Left boundary j = 0
                d_u_next[i[0] * NY + 0] = d_u_next[i[0] * NY + 1];
                // Right boundary j = NY-1
                d_u_next[i[0] * NY + (NY - 1)] = d_u_next[i[0] * NY + (NY - 2)];
            });
        }).wait();

        // Swap pointers to advance to the next step
        swap(d_u_prev, d_u_curr);
        swap(d_u_curr, d_u_next);

        // Optional: copy back to host and output every N steps
        // if(t % 100 == 0) {
        //     q.memcpy(u_curr.data(), d_u_curr, sizeof(double) * NX * NY).wait();
        //     cout << "Step " << t << " completed.\n";
        //     // Add saving or visualization code here
        // }
    }

    // Copy results back to host
    q.memcpy(u_curr.data(), d_u_curr, sizeof(double) * NX * NY).wait();


    std::cout << "{";
    for(int i = 0; i < 1; i++){
        //std::cout << "{";
        for(int j = 0; j < NY; j++){
            std::cout << u_curr[i * NX + j]  ;
            if (j < NY - 1) std::cout << ", ";
        }
        //std::cout << "}";
        //if (i < NX - 1) std::cout << ", ";
        
    }
    std::cout << "}" << std::endl;


    // Free device memory
    free(d_u_prev, q);
    free(d_u_curr, q);
    free(d_u_next, q);

    // Output some final result values as examples
    //cout << "Final temperature at center: " << u_curr[(NX/2)*NY + (NY/2)] << "\n";

    return 0;
}
