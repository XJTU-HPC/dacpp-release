#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

// Parameter settings
const double A = 1.0;  // Attraction coefficient
const double D = 0.1;  // Diffusion coefficient
const double dx = 0.1; // Spatial step size
const double dt = 0.01; // Time step size
const int N = 100;     // Number of spatial grid points
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

// Numerically solve the Fokker-Planck equation
void solveFokkerPlanck(std::vector<double>& p) {
    std::vector<double> new_p(N, 0.0); // Store distribution at next time step

    std::ofstream outFile("preference_distribution.csv");
    outFile << "TimeStep,";
    for (int i = 0; i < N; ++i) {
        outFile << "p(" << i * dx << "),";
    }
    outFile << std::endl;

    for (int t = 0; t < T; ++t) {
        // Update interior points, boundary conditions not considered for now
        for (int i = 1; i < N - 1; ++i) {
            double diffusion = D * (p[i + 1] - 2 * p[i] + p[i - 1]) / (dx * dx);
            double drift = -A * (p[i + 1] - p[i - 1]) / (2 * dx);
            new_p[i] = p[i] + dt * (diffusion + drift);
        }

        // Update distribution
        p = new_p;
        normalize(p); // Normalize distribution

        // Output current distribution
        if (t % 100 == 0) {
            //std::cout << "Time step: " << t << std::endl;
            outFile << t << ",";
            for (int i = 0; i < N; ++i) {
                outFile << p[i] << ",";
            }
            outFile << std::endl;
        }
    }
    for (int i = 0; i < N; ++i) {
        std::cout << p[i] << ",";
    }          

    outFile.close();
}

int main() {
    std::vector<double> p(N, 0.0); // Store user preference distribution

    // Initialize preference distribution
    initialize(p);

    // Numerically solve the Fokker-Planck equation
    solveFokkerPlanck(p);

    return 0;
}
