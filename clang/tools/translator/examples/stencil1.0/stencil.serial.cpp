#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

// Grid parameters
const int NX = 512;           // Number of grid points in x direction
const int NY = 512;           // Number of grid points in y direction
const float Lx = 10.0f;       // Length in x direction
const float Ly = 10.0f;       // Length in y direction
const float alpha = 0.01f;    // Thermal diffusion coefficient
const int TIME_STEPS = 1000;  // Number of time steps

// Save results to file
void save_to_file(const vector<float>& data, int nx, int ny, const string& filename) {
    ofstream file(filename);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            file << data[i * ny + j] << " ";
        }
        file << "\n";
    }
    file.close();
}

int main() {
    // Spatial step size
    float dx = Lx / (NX - 1);
    float dy = Ly / (NY - 1);

    // Stability condition
    float dt_stability = (dx * dx * dy * dy) / (2.0f * alpha * (dx * dx + dy * dy));
    float delta_t = 0.4f * dt_stability; // Stable time step

    cout << "Grid size: " << NX << "x" << NY << "\n";
    cout << "dx = " << dx << ", dy = " << dy << ", delta_t = " << delta_t << "\n";

    // Initialize temperature field
    vector<float> u_prev(NX * NY, 0.0f); // Previous step
    vector<float> u_curr(NX * NY, 0.0f); // Current step

    // Initial condition: Gaussian distribution at center
    int cx = NX / 2;
    int cy = NY / 2;
    float sigma = 1.0f;
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            float x = i * dx;
            float y = j * dy;
            u_curr[i * NY + j] = exp(-((x - Lx / 2.0f) * (x - Lx / 2.0f) + 
                                       (y - Ly / 2.0f) * (y - Ly / 2.0f)) / (2.0f * sigma * sigma));
        }
    }

    // Main iteration loop
    for (int t = 0; t < TIME_STEPS; ++t) {
        vector<float> u_next(NX * NY, 0.0f);

        // Update interior points
        for (int i = 1; i < NX - 1; ++i) {
            for (int j = 1; j < NY - 1; ++j) {
                float u_xx = (u_curr[(i + 1) * NY + j] - 2.0f * u_curr[i * NY + j] + 
                              u_curr[(i - 1) * NY + j]) / (dx * dx);
                float u_yy = (u_curr[i * NY + (j + 1)] - 2.0f * u_curr[i * NY + j] + 
                              u_curr[i * NY + (j - 1)]) / (dy * dy);

                u_next[i * NY + j] = u_curr[i * NY + j] + alpha * delta_t * (u_xx + u_yy);
            }
        }

        // Handle boundary conditions (adiabatic boundary: zero derivative)
        for (int j = 0; j < NY; ++j) {
            u_next[0 * NY + j] = u_next[1 * NY + j];              // Top boundary
            u_next[(NX - 1) * NY + j] = u_next[(NX - 2) * NY + j]; // Bottom boundary
        }
        for (int i = 0; i < NX; ++i) {
            u_next[i * NY + 0] = u_next[i * NY + 1];              // Left boundary
            u_next[i * NY + (NY - 1)] = u_next[i * NY + (NY - 2)]; // Right boundary
        }

        // Update current temperature field
        u_curr = u_next;

        // Optional: output progress every N steps
        if (t % 100 == 0) {
            cout << "Step " << t << " completed.\n";
        }
    }

    // Save final results
    save_to_file(u_curr, NX, NY, "final_temperature_serial.txt");

    cout << "Final temperature at center: " << u_curr[(NX / 2) * NY + (NY / 2)] << "\n";

    return 0;
}
