#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <fstream>

const int WIDTH = 10;       // Road segment length
const int TIME_STEPS = 20;  // Number of time steps
const double DELTA_T = 0.01; // Time step size
const double DELTA_X = 1.0;  // Spatial step size

// Flow function, considering the effect of density on flow
double q(double rho) {
    const double V_max = 30.0; // Maximum velocity
    const double rho_max = 50.0; // Maximum density
    return rho * V_max * (1 - rho / rho_max);
}

// Initialize density using a random distribution
void initializeDensity(std::vector<double>& rho) {
    for (int i = 0; i < WIDTH; ++i) {
        if (i < WIDTH / 4) {
            rho[i] = 40.0; // High density region
        } else if (i < 3 * WIDTH / 4) {
            rho[i] = 20.0; // Medium density region
        } else {
            rho[i] = 10.0; // Low density region
        }
    }
}


int main() {
    std::vector<double> rho(WIDTH, 0.0);
    std::vector<double> new_rho(WIDTH, 0.0);

    initializeDensity(rho);

    std::ofstream outputFile("density_output.csv");
    if (!outputFile) {
        std::cerr << "Unable to open file!" << std::endl;
        return 1;
    }

    // Write header
    outputFile << "Time Step";
    for (int x = 0; x < WIDTH; ++x) {
        outputFile << "," << x;
    }
    outputFile << std::endl;

    for (int t = 0; t < TIME_STEPS; ++t) {
        for (int x = 1; x < WIDTH - 1; ++x) {
            double flow_left = q(rho[x - 1]);
            double flow_right = q(rho[x]);
            double delta_rho = (DELTA_T / DELTA_X) * (flow_right - flow_left);
            
            // Update density and ensure no negative values
            new_rho[x] = rho[x] - delta_rho;

            // Ensure density is non-negative
            new_rho[x] = std::max(0.0, new_rho[x]);

            // Introduce diffusion effect to increase density uniformity
            if (x > 1 && x < WIDTH - 2) {
                new_rho[x] += 0.1 * (rho[x - 1] + rho[x + 1] - 2 * rho[x]); // Simple diffusion model
            }
        }

        // Handle boundary conditions
        new_rho[0] = new_rho[1]; // Left boundary: no traffic flow
        new_rho[WIDTH - 1] = new_rho[WIDTH - 2]; // Right boundary: no traffic flow
        
        // Update density values
        rho = new_rho;

        // Output density distribution for current time step to file
        outputFile << t; 
        for (int x = 0; x < WIDTH; ++x) {
            outputFile << "," << static_cast<int>(rho[x]);
        }
        outputFile << std::endl;
    }

    outputFile.close();
    return 0;
}
