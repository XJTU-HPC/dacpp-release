#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"
using namespace std;

namespace dacpp {
    typedef std::vector<std::any> list;
}

// =====================
// Global physical parameters
// =====================
const int NX = 8;
const int NY = 8;
const int NZ = 8;
const double Lx = 10.0;
const double Ly = 10.0;
const double Lz = 10.0;
const double alpha = 0.01;
const int TIME_STEPS = 100;

const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);
const double dz = Lz / (NZ - 1);

// 3D stability condition
const double dt_stability = 1.0 / (2.0 * alpha * (1.0/(dx*dx) + 1.0/(dy*dy) + 1.0/(dz*dz)));
const double delta_t = 0.4 * dt_stability;

// =====================
// Shell: 3D window mapping
// =====================
shell dacpp::list stencil3D_shell(
    dacpp::Tensor<double, 3>& matIn READ_WRITE, 
    dacpp::Tensor<double, 3>& matOut READ_WRITE
) {
    // Define sliding windows for three dimensions, size 3, stride 1
    dacpp::split sp1(3, 1), sp2(3, 1), sp3(3, 1);
    dacpp::index idx1, idx2, idx3;

    binding(sp1, idx1);
    binding(sp2, idx2);
    binding(sp3, idx3);

    // Mapping: 3x3x3 neighborhood of matIn -> corresponding grid point of matOut
    dacpp::list dataList{
        matIn[sp1][sp2][sp3],  // 3D neighborhood block
        matOut[idx1][idx2][idx3] // Target scalar
    };
    return dataList;
}

// =====================
// Calc: 7-point operator computation
// =====================
calc void stencil3D_calc(
    dacpp::Tensor<double, 3>& blockIn, 
    double* res
) {
    // In the 3x3x3 block, the center point is [1][1][1]
    // Neighbors are up, down, left, right, front, back
    double center = blockIn[1][1][1];
    
    double d2u_dx2 = (blockIn[2][1][1] - 2.0 * center + blockIn[0][1][1]) / (dx * dx);
    double d2u_dy2 = (blockIn[1][2][1] - 2.0 * center + blockIn[1][0][1]) / (dy * dy);
    double d2u_dz2 = (blockIn[1][1][2] - 2.0 * center + blockIn[1][1][0]) / (dz * dz);

    res[0] = center + alpha * delta_t * (d2u_dx2 + d2u_dy2 + d2u_dz2);
}

int main() {
    // 1. Initialize data
    vector<double> u_curr(NX * NY * NZ, 0.0);
    vector<double> u_next_data(NX * NY * NZ, 0.0);

    double cx = Lx / 2.0, cy = Ly / 2.0, cz = Lz / 2.0;
    double sigma = 1.0;
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                double x = i * dx, y = j * dy, z = k * dz;
                double dist_sq = (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz);
                u_curr[i * NY * NZ + j * NZ + k] = std::exp(-dist_sq / (2.0 * sigma * sigma));
            }
        }
    }

    // 2. Construct dacpp Tensor objects
    dacpp::Tensor<double, 3> matIn({NX, NY, NZ}, u_curr);
    dacpp::Tensor<double, 3> u_next_tensor({NX, NY, NZ}, u_next_data);
    
    // Compute only for the interior region (Slice)
    dacpp::Tensor<double, 3> matOut = u_next_tensor[{1, NX-1}][{1, NY-1}][{1, NZ-1}];

    // 3. Time step iteration
    for (int t = 0; t < TIME_STEPS; ++t) {
        // Execute parallel Stencil computation
        stencil3D_shell(matIn, matOut) <-> stencil3D_calc;

        // Handle boundary conditions and sync back to matIn (following 2D dacpp style)
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                for (int k = 0; k < NZ; k++) {
                    // Update interior points from matOut
                    if (i > 0 && i < NX - 1 && j > 0 && j < NY - 1 && k > 0 && k < NZ - 1) {
                        matIn[i][j][k] = matOut[i - 1][j - 1][k - 1];
                    }
                }
            }
        }
        
        // Boundary handling (6 faces)
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                matIn[0][j][k] = matIn[1][j][k];
                matIn[NX-1][j][k] = matIn[NX-2][j][k];
            }
        }
        for (int i = 0; i < NX; i++) {
            for (int k = 0; k < NZ; k++) {
                matIn[i][0][k] = matIn[i][1][k];
                matIn[i][NY-1][k] = matIn[i][NY-2][k];
            }
        }
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                matIn[i][j][0] = matIn[i][j][1];
                matIn[i][j][NZ-1] = matIn[i][j][NZ-2];
            }
        }
    }

    // 4. Output results
    // std::cout << "{";
    // for (int i = 0; i < NX; i++) {
    //     std::cout << "{";
    //     for (int j = 0; j < NY; j++) {
    //         std::cout << "{";
    //         for (int k = 0; k < NZ; k++) {
    //             std::cout << matIn[i][j][k];
    //             if (k < NZ - 1) std::cout << ", ";
    //         }
    //         std::cout << "}";
    //         if (j < NY - 1) std::cout << ", ";
    //     }
    //     std::cout << "}";
    //     if (i < NX - 1) std::cout << ", ";
    // }
    // std::cout << "}" << std::endl;

    matIn.print();
    return 0;
}