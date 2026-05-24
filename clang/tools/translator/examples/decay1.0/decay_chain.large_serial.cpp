#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

int main() {
    size_t numIsotopes = 10000; // Number of isotopes
    double dt = 0.1;         // Time step size
    double T = 5.0;         // Total time
    size_t numSteps = static_cast<size_t>(T / dt); // Total number of steps

    // Initialize decay constants and initial quantities
    std::vector<double> lambdas(numIsotopes);
    std::vector<double> N0s(numIsotopes, 1000.0);  // Initial quantity is 1000
    for (size_t i = 0; i < numIsotopes; ++i) {
        lambdas[i] = 0.01 + 0.01 * i;  // lambda increases linearly [0.01, 0.1]
    }

    // Store data for all time steps
    std::vector<std::vector<double>> results(numSteps + 1, std::vector<double>(numIsotopes));
    results[0] = N0s; // Initial values

    // Compute the decay process for isotopes
    for (size_t t = 1; t <= numSteps; ++t) {
        for (size_t i = 0; i < numIsotopes; ++i) {
            results[t][i] = results[t - 1][i] * std::exp(-lambdas[i] * dt);
        }
    }

    // Output in "{{...}, {...}, ...}" format
    std::cout << "{";
    for (size_t t = 1; t <= 1; ++t) {
        //std::cout << "{";
        for (size_t i = 0; i < numIsotopes; ++i) {
            std::cout << results[t][i];
            if (i < numIsotopes - 1) std::cout << ", ";
        }
        //std::cout << "}";
        //if (t < numSteps) std::cout << ", ";
    }
    std::cout << "}" << std::endl;
    
    return 0;
}
