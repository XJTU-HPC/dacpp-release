#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

using namespace std;
using Complex = complex<double>;  // Complex number type alias


// Discrete Fourier Transform (DFT)
void dft(const vector<Complex>& input, vector<Complex>& output) {
    int N = input.size();
    output.resize(N);

    // DFT formula: X[k] = Σ (x[n] * e^(-2πi * k * n / N)), k=0 to N-1
    for (int k = 0; k < N; ++k) {
        //std::cout << "k=" << k << std::endl;
        Complex sum(0, 0);
        for (int n = 0; n < N; ++n) {
            double angle = -2.0 * M_PI * k * n / N;
            Complex W_n(cos(angle), sin(angle));  // Twiddle factor
            sum += input[n] * W_n;  // Accumulate each term
        }
        output[k] = sum;  // Fourier transform result
    }
}


int main() {
    // Define an input signal (complex sequence of length 8)
    int N = 8;
    vector<Complex> input(N);

    // Initialize input data (can be any time-domain signal)
    for (int i = 0; i < N; ++i) {
        input[i] = Complex(i, 0);  // Fill with complex data, simple example here
    }

    // Output original data
    //cout << "Original data (time domain):" << endl;
    // for (const auto& val : input) {
    //     cout << val << endl;
    // }

    // Compute the discrete Fourier transform
    vector<Complex> output;
    dft(input, output);

    // Output the Fourier transformed data (frequency domain)
    //cout << "\nFourier transformed data (frequency domain):" << endl;
    std::cout << "{";
    for (size_t i = 0; i < output.size(); ++i) {
        cout << "(" << output[i].real() << "," << output[i].imag() << ")";
        if (i != output.size() - 1) {
            cout << ", ";  // Avoid trailing comma after the last element
        }
    }
    std::cout << "}\n";

    return 0;
}
