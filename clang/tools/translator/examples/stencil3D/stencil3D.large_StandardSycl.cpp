#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace sycl;
using namespace std;

// 3D grid parameters
const int NX = 256;           // Number of grid points in x direction
const int NY = 256;           // Number of grid points in y direction
const int NZ = 256;           // Number of grid points in z direction
const double Lx = 10.0;
const double Ly = 10.0;
const double Lz = 10.0;
const double alpha = 0.01;    // Thermal diffusivity
const int TIME_STEPS = 100;

int main() {
    queue q{property::queue::in_order()};

    // Spatial step sizes
    double dx = Lx / (NX - 1);
    double dy = Ly / (NY - 1);
    double dz = Lz / (NZ - 1);

    // 3D stability condition: dt < 1 / (2 * alpha * (1/dx^2 + 1/dy^2 + 1/dz^2))
    double dt_stability = 1.0 / (2.0 * alpha * (1.0/(dx*dx) + 1.0/(dy*dy) + 1.0/(dz*dz)));
    double delta_t = 0.4 * dt_stability; 

    size_t num_elements = NX * NY * NZ;
    
    // Allocate device memory using USM
    double* d_u_curr = malloc_device<double>(num_elements, q);
    double* d_u_next = malloc_device<double>(num_elements, q);
    vector<double> h_u(num_elements, 0.0);

    // 1. Initialize temperature field (3D Gaussian heat source)
    double cx = Lx / 2.0, cy = Ly / 2.0, cz = Lz / 2.0;
    double sigma = 1.0;
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                double x = i * dx, y = j * dy, z = k * dz;
                double dist_sq = (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz);
                h_u[i * NY * NZ + j * NZ + k] = std::exp(-dist_sq / (2.0 * sigma * sigma));
            }
        }
    }

    q.memcpy(d_u_curr, h_u.data(), sizeof(double) * num_elements).wait();

    // 2. Time step iteration
    for (int t = 0; t < TIME_STEPS; ++t) {
        // Interior point computation (7-point Stencil)
        q.submit([&](handler &h) {
            h.parallel_for(range<3>(NX - 2, NY - 2, NZ - 2), [=](id<3> idx) {
                int i = idx[0] + 1;
                int j = idx[1] + 1;
                int k = idx[2] + 1;

                // 3D index mapping
                auto get_idx = [&](int x, int y, int z) {
                    return x * NY * NZ + y * NZ + z;
                };

                int curr = get_idx(i, j, k);
                
                double d2u_dx2 = (d_u_curr[get_idx(i+1, j, k)] - 2.0 * d_u_curr[curr] + d_u_curr[get_idx(i-1, j, k)]) / (dx * dx);
                double d2u_dy2 = (d_u_curr[get_idx(i, j+1, k)] - 2.0 * d_u_curr[curr] + d_u_curr[get_idx(i, j-1, k)]) / (dy * dy);
                double d2u_dz2 = (d_u_curr[get_idx(i, j, k+1)] - 2.0 * d_u_curr[curr] + d_u_curr[get_idx(i, j, k-1)]) / (dz * dz);

                d_u_next[curr] = d_u_curr[curr] + alpha * delta_t * (d2u_dx2 + d2u_dy2 + d2u_dz2);
            });
        });

        // 3. Boundary condition handling (simplified: mirror or fixed boundaries, using neighbor copy to simulate adiabatic)
        // Process 6 faces
        q.submit([&](handler &h) {
            h.parallel_for(range<2>(NY, NZ), [=](id<2> idx) {
                int j = idx[0], k = idx[1];
                d_u_next[0 * NY * NZ + j * NZ + k] = d_u_next[1 * NY * NZ + j * NZ + k]; // X direction minimum boundary
                d_u_next[(NX-1) * NY * NZ + j * NZ + k] = d_u_next[(NX-2) * NY * NZ + j * NZ + k]; // X direction maximum boundary
            });
        });
        q.submit([&](handler &h) {
            h.parallel_for(range<2>(NX, NZ), [=](id<2> idx) {
                int i = idx[0], k = idx[1];
                d_u_next[i * NY * NZ + 0 * NZ + k] = d_u_next[i * NY * NZ + 1 * NZ + k]; // Y direction minimum boundary
                d_u_next[i * NY * NZ + (NY-1) * NZ + k] = d_u_next[i * NY * NZ + (NY-2) * NZ + k]; // Y direction maximum boundary
            });
        });
        q.submit([&](handler &h) {
            h.parallel_for(range<2>(NX, NY), [=](id<2> idx) {
                int i = idx[0], j = idx[1];
                d_u_next[i * NY * NZ + j * NZ + 0] = d_u_next[i * NY * NZ + j * NZ + 1]; // Z direction minimum boundary
                d_u_next[i * NY * NZ + j * NZ + (NZ-1)] = d_u_next[i * NY * NZ + j * NZ + (NZ-2)]; // Z direction maximum boundary
            });
        });

        q.wait();
        // Swap pointers
        std::swap(d_u_curr, d_u_next);
    }

    // Copy results back to host
    q.memcpy(h_u.data(), d_u_curr, sizeof(double) * num_elements).wait();

    // --- Nested output following 3D matrix multiplication pattern ---
    std::cout << "{";
    for (int i = 0; i < NX; i++) {
        std::cout << "{";
        for (int j = 0; j < NY; j++) {
            std::cout << "{";
            for (int k = 0; k < NZ; k++) {
                // Index logic: i is outer (X), j is middle (Y), k is inner (Z)
                std::cout << h_u[i * NY * NZ + j * NZ + k];
                if (k < NZ - 1) std::cout << ", ";
            }
            std::cout << "}";
            if (j < NY - 1) std::cout << ", ";
        }
        std::cout << "}";
        if (i < NX - 1) std::cout << ", ";
    }
    std::cout << "}" << std::endl;

    // cout << "3D Stencil computation completed." << std::endl;
    // cout << "Center value: " << h_u[(NX/2) * NY * NZ + (NY/2) * NZ + (NZ/2)] << std::endl;

    free(d_u_curr, q);
    free(d_u_next, q);

    return 0;
}