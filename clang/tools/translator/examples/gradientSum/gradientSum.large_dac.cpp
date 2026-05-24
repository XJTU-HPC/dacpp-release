#include <iostream>
#include <vector>
#include <random>
#include <any>
#include "ReconTensor.h"

using namespace std;

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int NUM_NEURONS = 1024;   // Number of neurons (layer width)
const int INPUT_SIZE  = 1024;   // Number of inputs per neuron

shell dacpp::list gradSumShell(dacpp::Matrix<float>& matGrads READ_WRITE,
                               dacpp::Matrix<float>& matNeuronSum READ_WRITE) {
    dacpp::index idx1, idx2;
    dacpp::list dataList{matGrads[{}][idx1], matNeuronSum[idx1][idx2]};
    return dataList;
}

// -----------------------------
// DAC calc function: reduction addition
// -----------------------------
calc void gradSum(dacpp::Vector<float>& grads, dacpp::Vector<float>& neuronSum) {
    int sum=0;
    for (int j = 0; j < INPUT_SIZE; ++j) {
        sum += grads[j];   // Reduction addition
    }
    neuronSum[0]=sum;
}

// -----------------------------
// Main function
// -----------------------------
int main() {
    // Initialize gradient matrix (simulated random gradients)
    vector<float> host_grads(NUM_NEURONS * INPUT_SIZE);
    mt19937 gen(42);
    uniform_real_distribution<float> dist(-0.1f, 0.1f);
    // for (auto &v : host_grads) v = dist(gen);
    for(int i=0;i<NUM_NEURONS;i++){
        for(int j=0;j<INPUT_SIZE;j++){
            host_grads[i*INPUT_SIZE+j]=j;
        }
    }
    // DAC Tensor initialization
    dacpp::Matrix<float> matGrads({NUM_NEURONS, INPUT_SIZE}, host_grads);
    vector<float> host_neuron_sum(NUM_NEURONS, 0.0f);
    dacpp::Matrix<float> matNeuronSum({NUM_NEURONS, 1}, host_neuron_sum);

    // Execute DAC shell -> calc
    gradSumShell(matGrads, matNeuronSum) <-> gradSum;

    // Output results
    std::cout << "First 5 neuron gradient sums:\n";
    for (size_t i = 0; i < std::min(5, NUM_NEURONS) ; ++i)
        std::cout << matNeuronSum[i][0] << " ";
    std::cout << std::endl;

    return 0;
}