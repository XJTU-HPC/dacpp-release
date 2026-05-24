#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace sycl;
using namespace std;

// Grid parameters
const int NX = 8;    // Number of grid points in x direction
const int NY = 8;    // Number of grid points in y direction
const double Lx = 10.0f; // Length in x direction
const double Ly = 10.0f; // Length in y direction
const double c = 1.0f;   // Wave speed
const int TIME_STEPS = 10; // Number of time steps


int main() {
    // Grid spacing
    double dx = Lx / (NX - 1);
    double dy = Ly / (NY - 1);
    
    // CFL condition
    double dt = 0.5f * std::fmin(dx, dy) / c; // Satisfy stability condition
    
    // Initialize wave field
    vector<double> u_prev(NX * NY, 0.0f); // Previous step
    vector<double> u_curr(NX * NY, 0.0f); // Current step
    vector<double> u_next(NX * NY, 0.0f); // Next step
    
    // Initial condition: e.g., a Gaussian pulse
    int cx = NX / 2;
    int cy = NY / 2;
    double sigma = 0.5f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            double x = i * dx;
            double y = j * dy;
            u_prev[i * NY + j] = std::exp(-((x - Lx/2)*(x - Lx/2) + (y - Ly/2)*(y - Ly/2)) / (2 * sigma * sigma));
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
            h.parallel_for<class wave_compute>(
                range<2>{NX - 2, NY - 2},
                [=](id<2> idx) {
                    int i = idx[0] + 1;
                    int j = idx[1] + 1;
                    // Second-order central difference
                    double u_xx = (d_u_curr[(i+1) * NY + j] - 2.0f * d_u_curr[i * NY + j] + d_u_curr[(i-1) * NY + j]) / (dx * dx);
                    double u_yy = (d_u_curr[i * NY + (j+1)] - 2.0f * d_u_curr[i * NY + j] + d_u_curr[i * NY + (j-1)]) / (dy * dy);
                    
                    // Explicit update formula
                    d_u_next[i * NY + j] = 2.0f * d_u_curr[i * NY + j] - d_u_prev[i * NY + j] + (c * c) * (u_xx + u_yy) * dt * dt;
                }
            );
        });
        
        // Handle boundary conditions (simple fixed boundary, wave value = 0)
        q.submit([&](handler &h) {
            // Left and right boundaries
            h.parallel_for<class boundary_left_right>(range<1>{NX}, [=](id<1> i) {
                d_u_next[i[0] * NY + 0] = 0.0f;
                d_u_next[i[0] * NY + (NY-1)] = 0.0f;
            });
        });
        
        q.submit([&](handler &h) {
            // Top and bottom boundaries
            h.parallel_for<class boundary_up_down>(range<1>{NY}, [=](id<1> j) {
                d_u_next[0 * NY + j] = 0.0f;
                d_u_next[(NX-1) * NY + j] = 0.0f;
            });
        });
        
        q.wait();
        
        // Swap pointers
        swap(d_u_prev, d_u_curr);
        swap(d_u_curr, d_u_next);
        
        // Optional: copy back to host and output at regular intervals
        // if(t % 100 == 0) {
        //     q.memcpy(u_curr.data(), d_u_curr, sizeof(double) * NX * NY).wait();
        //     cout << "Step " << t << " completed.\n";
        //     // Add saving or visualization code here
        // }
    }
    
    // Copy results back to host
    q.memcpy(u_curr.data(), d_u_curr, sizeof(double) * NX * NY).wait();
    
    std::cout << "{";
    for(int i = 0; i < NX; i++){
        std::cout << "{";
        for(int j = 0; j < NY; j++){
            std::cout << u_curr[i * NX + j]  ;
            if (j < NY - 1) std::cout << ", ";
        }
        std::cout << "}";
        if (i < NX - 1) std::cout << ", ";
        
    }
    std::cout << "}" << std::endl;

    // Free device memory
    free(d_u_prev, q);
    free(d_u_curr, q);
    free(d_u_next, q);
    
    // Output some values of the final result as an example
    //cout << u_curr[(NX/2)*NY + (NY/2)] << "\n";
    
    return 0;
}
