#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <any>
#include <queue>
#include "ReconTensor.h"
namespace dacpp {
    typedef std::vector<std::any> list;
}
// Parameter settings
const double A = 1.0;  // Attraction coefficient
const double D = 0.1;  // Diffusion coefficient
const double dx = 0.1; // Spatial step size
const double dt = 0.01; // Time step size
const int N = 150;     // Number of spatial grid points
const int T = 1000;    // Number of time steps

// Initialize user preference distribution
void initialize(std::vector<double>& p) {
    for (int i = 0; i < N; ++i) {
        // Assume initial preference follows a Gaussian distribution
        double x = i * dx;
        p[i] = std::exp(-std::pow(x - 5.0, 2) / 2.0); // Initial preference distribution centered at x=5
    }
}

// Normalization function
void normalize(dacpp::Vector<double>& p) {
    double sum = 0.0;
    for (int i = 0;i < N-2; i++) {
        sum += p[i];
    }
    for (int i = 0;i < N-2; i++) {
        p[i] /= sum; // Normalize
    }
}

shell dacpp::list mdp_shell( dacpp::Vector<double>& p READ_WRITE, dacpp::Vector<double>& new_p READ_WRITE){
    dacpp::index idx;
    dacpp::split sp(3,1);
    binding(idx, sp);
    dacpp::list dataList{p[sp],new_p[idx]};
    return dataList;
}

calc void mdp(dacpp::Vector<double>& p, double* new_p){
    double diffusion = D * (p[2] - 2 * p[1] + p[0]) / (dx * dx) ;
    double drift = (-A) * (p[2] - p[0]) / (2 * dx);
    new_p[0] = p[1] + dt * (diffusion+ drift);
}

// Numerically solve the Fokker-Planck equation

int main() {
    std::vector<double> p1(N, 0.0); // Store user preference distribution
    // Initialize preference distribution
    initialize(p1);
    // Numerically solve the Fokker-Planck equation
    std::vector<double> new_p1(N-2, 0.0); // Store distribution at next time step
    dacpp::Vector<double> p(p1);
    dacpp::Vector<double> new_p(new_p1);
    for (int t = 0; t < T; ++t) {
        mdp_shell(p, new_p) <->  mdp;
        //normalize(new_p); // Normalize distribution
        // Update distribution
        for(int i = 0; i <= N-3; i++){
            p[i+1] = new_p[i];
        }
        // Set boundary conditions
        //p[0] = 0.0;
        //p[N - 1] = 0.0;
        
    }
    std::cout << p[2] << std::endl;
    //p.print();
    return 0;
}
