#include <iostream>
#include <vector>
#include <complex>
#include "ReconTensor.h"
#include <cmath>

namespace dacpp {
    typedef std::vector<std::any> list;
}
using namespace std;


// Global variable definitions
const int row_count = 8, col_count = 8, max_iterations = 1000;
vector<complex<float>> complex_points;  // One-dimensional vector representing complex points
vector<int> mandelbrot_flags;           // One-dimensional array indicating whether each point belongs to the Mandelbrot set
int total_points = 0;                   // Total number of points
int mandelbrot_count = 0;               // Number of points belonging to the Mandelbrot set

// Initialize complex points vector
void InitializeComplexPoints() {
    total_points = row_count * col_count;  // Total number of points
    complex_points.resize(total_points);

    for (int i = 0; i < row_count; ++i) {
        for (int j = 0; j < col_count; ++j) {
            int index = i * col_count + j;  // One-dimensional vector index
            float real = -1.5f + (i * (2.0f / row_count));  // Map row index to real part
            float imag = -1.0f + (j * (2.0f / col_count));  // Map column index to imaginary part
            complex_points[index] = complex<float>(real, imag);
        }
    }
}


shell dacpp::list MANDEL(const dacpp::Vector<complex<float>>& complex_points,
                        dacpp::Vector<int>& mandelbrot_flags) {
    dacpp::index i;
    dacpp::list dataList{complex_points[i], mandelbrot_flags[i]};
    return dataList;
}

calc void mandel(complex<float>* complex_points,
                int* mandelbrot_flags) {
    const complex<float>& c = complex_points[0];
    complex<float> z = 0;
    int iterations = 0;
    for (int i = 0; i < max_iterations; ++i) {

        if (std::sqrt(z.real()*z.real() + z.imag()*z.imag()) > 2.0f) {
            iterations = i;
            break;
        }
        // Stop iteration if out of bounds
        z = z * z + c;
        iterations = max_iterations;
    }

    if (iterations == max_iterations) {
        mandelbrot_flags[0] = 1;  // Set to 1, indicating it belongs to the Mandelbrot set
    }


}


// Print statistics
void PrintStats() {
    cout << "Mandelbrot Set Statistics:\n";
    cout << "Total points: " << total_points << "\n";
    cout << "Points in the Mandelbrot set: " << mandelbrot_count << "\n";
}

int main() {
    // Initialize complex points vector
    InitializeComplexPoints();

    // Compute the Mandelbrot set
    mandelbrot_flags.resize(total_points, 0);  // Initialize one-dimensional array to 0

    dacpp::Vector<complex<float>> complex_points_tensor(complex_points);
    dacpp::Vector<int> mandelbrot_flags_tensor(mandelbrot_flags);


    MANDEL(complex_points_tensor, mandelbrot_flags_tensor) <-> mandel;

    // Count the number of 1s in the array
    mandelbrot_count = 0;
    for (int i = 0; i < total_points; i++){
        if (mandelbrot_flags_tensor[i] == 1) mandelbrot_count++;
    }

    // Print statistics
    PrintStats();

    return 0;
}
