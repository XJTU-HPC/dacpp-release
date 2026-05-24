#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>

const int WIDTH = 100;       // Road segment length
const int TIME_STEPS = 200;  // Number of time steps
const double DELTA_T = 0.01; // Time step size
const double DELTA_X = 1.0;  // Spatial step size


// Flow function, considering the effect of density on flow
double q1(double rho) {
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

    // SYCL queue for selecting device and executing parallel tasks
    sycl::queue q;

    for (int t = 0; t < TIME_STEPS; ++t) {
        // Use SYCL buffer and parallel_for to update density in parallel
        {
            sycl::buffer<double, 1> rho_buf(rho.data(), sycl::range<1>(WIDTH));
            sycl::buffer<double, 1> new_rho_buf(new_rho.data(), sycl::range<1>(WIDTH));

            q.submit([&](sycl::handler& h) {
                // Get read/write access for rho and new_rho
                auto rho_acc = rho_buf.get_access<sycl::access::mode::read>(h);
                auto new_rho_acc = new_rho_buf.get_access<sycl::access::mode::write>(h);

                h.parallel_for(sycl::range<1>(WIDTH - 2), [=](sycl::id<1> i) {
                    int x = i[0] + 1; // Skip boundary
                    double flow_left = q1(rho_acc[x - 1]);
                    double flow_right = q1(rho_acc[x]);
                    double delta_rho = (DELTA_T / DELTA_X) * (flow_right - flow_left);

                    new_rho_acc[x] = rho_acc[x] - delta_rho;
                    new_rho_acc[x] = std::max(0.0, new_rho_acc[x]);

                    // Diffusion effect
                    // if (x > 1 && x < WIDTH - 2) {
                    //     new_rho_acc[x] += 0.1 * (rho_acc[x - 1] + rho_acc[x + 1] - 2 * rho_acc[x]);
                    // }
                });
            });
        }

        // Handle boundary conditions
        new_rho[0] = new_rho[1]; // Left boundary: no traffic flow
        new_rho[WIDTH - 1] = new_rho[WIDTH - 2]; // Right boundary: no traffic flow

        // Update density values
        rho = new_rho;

        // Output density distribution for current time step
        //std::cout << "Time Step " << t << ": ";
        //for (int x = 0; x < WIDTH; ++x) {
            //std::cout << static_cast<int>(rho[x]) << " ";
        //}
        //std::cout << std::endl;
    }
    for (int x = 15; x < 16; ++x) {
        std::cout << static_cast<int>(rho[x]) << std::endl;
    }

    return 0;
}
