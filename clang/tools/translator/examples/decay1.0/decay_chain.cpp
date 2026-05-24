#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

// Calculate the quantity of each isotope at time t
void calculateDecay(const std::vector<double>& lambdas, const std::vector<double>& N0s, double dt, double T, size_t numOutputSteps) {
    size_t numIsotopes = lambdas.size(); // Number of isotopes
    std::vector<std::vector<double>> A(numIsotopes);  // Store the quantity of each isotope at different time points
    std::vector<double> time;  // Time series
    double t = 0;

    // Serial computation of the decay process for each isotope
    for (size_t i = 0; i < numIsotopes; ++i) {
        double N = N0s[i];  // Initial quantity of the current isotope
        double lambda = lambdas[i];  // Decay constant of the current isotope
        std::vector<double> local_A;  // Storage for the quantity of the current isotope over time

        // Perform decay calculation for the current isotope
        while (t <= T) {
            local_A.push_back(N * exp(-lambda * t));  // Calculate the quantity after decay
            t += dt;
        }

        // Store the result for the current isotope
        A[i] = local_A;
        t = 0;  // Reset time t for the next isotope to use the correct time step
    }

    // Compute the time series
    t = 0;
    while (t <= T) {
        time.push_back(t);
        t += dt;
    }

    // Output only a subset of time steps (e.g., first 5 time steps)
    size_t outputStepInterval = time.size() / numOutputSteps; // Calculate output interval
    std::cout << "Time ";
    // Output isotope labels (A1, A2, A3...)
    for (size_t i = 0; i < numIsotopes; ++i) {
        std::cout << "A" << i + 1 << " ";
    }
    std::cout << "\n";

    // Output data for each time step (partial time steps)
    for (size_t i = 0; i < numOutputSteps; ++i) {
        size_t timeIndex = i * outputStepInterval;
        std::cout << time[timeIndex];
        for (size_t j = 0; j < numIsotopes; ++j) {
            std::cout << " " << A[j][timeIndex];
        }
        std::cout << "\n";
    }
}

int main() {
    size_t numIsotopes = 10000; // Set a large number of isotopes (e.g., 10000)

    // Randomly generate decay constants and initial quantities
    std::vector<double> lambdas(numIsotopes);
    std::vector<double> N0s(numIsotopes, 1000.0);  // Initial quantity is 1000

    // Randomly initialize decay constants (e.g., lambda between 0.01 and 0.2)
    for (size_t i = 0; i < numIsotopes; ++i) {
        lambdas[i] = 0.01 + static_cast<double>(rand()) / (RAND_MAX / (0.2 - 0.01));  // lambda range [0.01, 0.2]
    }

    double dt = 0.1;       // Time step size
    double T = 50.0;       // Total time
    size_t numOutputSteps = 10; // Number of output time steps

    calculateDecay(lambdas, N0s, dt, T, numOutputSteps);

    return 0;
}
