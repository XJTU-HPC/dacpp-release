#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include "ReconTensor.h"
// Tensors inside for loops need array2Tensor after translation
namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;
using Complex = std::complex<double>;  // Complex number type alias
const int N = 8;



shell dacpp::list DFT(const dacpp::Vector<std::complex<double>>& input,
                        dacpp::Vector<std::complex<double>>& output, 
                    const dacpp::Vector<int>& vec) {
    dacpp::index i;
    dacpp::list dataList{input[{}], output[i], vec[i]};
    return dataList;
}

calc void dft(std::complex<double>* input,
                std::complex<double>* output, 
                int* vec) {
    Complex sum(0, 0);
    for (int n = 0; n < N; ++n) {
        double angle = -2.0 * M_PI * vec[0] * n / N;
        Complex W_n(std::cos(angle), std::sin(angle));  // Twiddle factor
        sum += input[n] * W_n;  // Accumulate each term
    }
    output[0] = sum;  // Fourier transform result
}

// Discrete Fourier Transform (DFT)
void dftfunc(const vector<std::complex<double>>& input, vector<std::complex<double>>& output) {
    int N = input.size();
    output.resize(N);

    std::vector<int> vec(N);

    // Initialize vector using for loop, elements from 0 to N-1
    for (int i = 0; i < N; ++i) {
        vec[i] = i;  // Assign values 0 to N-1
    }
    dacpp::Vector<int> vec_tensor(vec);
    dacpp::Vector<std::complex<double>> input_tensor(input);
    dacpp::Vector<std::complex<double>> output_tensor(output);

    // DFT formula: X[k] = Σ (x[n] * e^(-2πi * k * n / N)), k=0 to N-1
    DFT(input_tensor, output_tensor, vec_tensor) <-> dft;
    output_tensor.print();
}

int main() {
    // Define an input signal (complex sequence of length 8)

    vector<std::complex<double>> input(N);

    // Initialize input data (can be any time-domain signal)
    for (int i = 0; i < N; ++i) {
        input[i] = Complex(i, 0);  // Fill with complex data, simple example here
    }

    // Compute the discrete Fourier transform
    vector<Complex> output(N);
    int N = input.size();
    output.resize(N);

    std::vector<int> vec(N);

    // Initialize vector using for loop, elements from 0 to N-1
    for (int i = 0; i < N; ++i) {
        vec[i] = i;  // Assign values 0 to N-1
    }
    dacpp::Vector<int> vec_tensor(vec);
    dacpp::Vector<std::complex<double>> input_tensor(input);
    dacpp::Vector<std::complex<double>> output_tensor(output);

    // DFT formula: X[k] = Σ (x[n] * e^(-2πi * k * n / N)), k=0 to N-1
    DFT(input_tensor, output_tensor, vec_tensor) <-> dft;
    output_tensor.print();

    return 0;
}