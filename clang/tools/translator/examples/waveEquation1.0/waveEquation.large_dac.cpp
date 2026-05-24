#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"
using namespace std;

namespace dacpp {
    typedef std::vector<std::any> list;
}
// Grid parameters
const int NX = 8192;    // Number of grid points in x direction
const int NY = 8192;    // Number of grid points in y direction
const double Lx = 10.0f; // Length in x direction
const double Ly = 10.0f; // Length in y direction
const double c = 1.0f;   // Wave speed
const int TIME_STEPS = 1000; // Number of time steps
// Grid spacing
const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);


// CFL condition
const double dt = 0.5f * std::fmin(dx, dy) / c; // Satisfy stability condition

shell dacpp::list waveEqShell( dacpp::Matrix<double>& matCur READ_WRITE, 
                               dacpp::Matrix<double>& matPrev READ_WRITE, 
                               dacpp::Matrix<double>& matNext WRITE){
    dacpp::split sp1(3, 1), sp2(3, 1);
    dacpp::index idx1, idx2;
    binding(sp1, idx1);
    binding(sp2, idx2);
    dacpp::list dataList{matCur[sp1][sp2], matPrev[idx1][idx2], matNext[idx1][idx2]};
    return dataList;
}

calc void waveEq(dacpp::Matrix<double>& cur, double* prev, double* next) {
    double dt = 0.5f * std::fmin(dx, dy) / c; // Satisfy stability condition
    double u_xx = (cur[2][1] - 2.0f * cur[1][1] + cur[0][1])/ (dx * dx);
    double u_yy = (cur[1][2] - 2.0f * cur[1][1] + cur[1][0])/ (dy * dy);
    next[0]=2.0f*cur[1][1]-prev[0]+(c * c)*dt*dt*(u_xx+u_yy);
}

int main() {
    // Initialize wave field
    vector<double> u_prev(NX * NY, 0.0f); // Previous step
    vector<double> u_curr(NX * NY, 0.0f);  // Current step
    vector<double> u_next(NX * NY, 0.0f);  // Current step

    // Initial condition: e.g., a Gaussian pulse
    int cx = NX / 2;
    int cy = NY / 2;
    double sigma = 0.5f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            double x = i * dx;
            double y = j * dy;
            u_prev[i*NX+j] = std::exp(-((x - Lx/2)*(x - Lx/2) + (y - Ly/2)*(y - Ly/2)) / (2 * sigma * sigma));
        }
    }

    dacpp::Matrix<double> matCur({NX, NY}, u_curr);
    dacpp::Matrix<double> u_prev_tensor({NX, NY}, u_prev);
    dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);
    dacpp::Matrix<double> matPrev = u_prev_tensor[{1,NX-1}][{1,NY-1}];
    dacpp::Matrix<double> matNext = u_next_tensor[{1,NX-1}][{1,NY-1}];
    
    for(int i = 0;i < TIME_STEPS; i++) {
        waveEqShell(matCur, matPrev, matNext) <-> waveEq;
        for (int i = 1; i <= NX-2; i++) {
            for(int j = 1; j <=NY-2; j++){
                matPrev[i-1][j-1]=matCur[i][j];
            }
        }

        for (int i = 1; i <= NX-2; i++) {
            for(int j = 1; j <=NY-2; j++){
                matCur[i][j]=matNext[i-1][j-1];
            }
        }
        // Handle boundary conditions (adiabatic boundary: zero derivative)
        for (int i = 0; i <= NX-1; ++i) {       
            matCur[i][NY-1]=0;
            matCur[i][0]=0;
        }
        for (int j = 0; j <= NY-1; ++j) {
            matCur[NX - 1][j]=0;
            matCur[0][j]=0;
             // Bottom boundary
        }
    }
    //
    matCur.print(); 
    return 0;
}