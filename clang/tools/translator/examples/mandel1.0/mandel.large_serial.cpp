#include <iostream>
#include <vector>
#include <complex>

using namespace std;



// Global variable definitions
int row_count = 16, col_count = 16, max_iterations = 1000;
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

// Compute the number of iterations for a point in the Mandelbrot set
int ComputePoint(const complex<float>& c) {
    complex<float> z = 0;
    for (int i = 0; i < max_iterations; ++i) {
        if (abs(z) > 2.0f) return i;  // Stop iteration if out of bounds
        z = z * z + c;
    }
    return max_iterations;  // If it did not diverge, return the maximum iteration count
}

// Compute the Mandelbrot set
void ComputeMandelbrot() {
    mandelbrot_flags.resize(total_points, 0);  // Initialize one-dimensional array to 0

    for (int index = 0; index < total_points; ++index) {
        const complex<float>& c = complex_points[index];  // Get complex point
        int iterations = ComputePoint(c);
        if (iterations == max_iterations) {
            mandelbrot_flags[index] = 1;  // Set to 1, indicating it belongs to the Mandelbrot set
        }
    }

    // Count the number of 1s in the array
    mandelbrot_count = 0;
    for (int flag : mandelbrot_flags) {
        if (flag == 1) mandelbrot_count++;
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
    ComputeMandelbrot();

    // Print statistics
    PrintStats();

    return 0;
}
