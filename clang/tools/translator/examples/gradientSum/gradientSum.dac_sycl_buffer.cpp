#include <iostream>
#include <vector>
#include <random>
#include <any>
#include "ReconTensor.h"

using namespace std;

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int NUM_NEURONS = 8;   // Number of neurons (layer width)
const int INPUT_SIZE  = 8;   // Number of inputs per neuron



// -----------------------------
// DAC calc function: reduction addition
// -----------------------------


// -----------------------------
// Main function
// -----------------------------
#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void gradSum(float* grads,float* neuronSum,int grads_0,int grads_1,int neuronSum_0,int neuronSum_1,int grads_0_shape,int grads_1_shape,int neuronSum_0_shape,int neuronSum_1_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_grads_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_neuronSum_acc) {
    int sum = 0;
    for (int j = 0; j < INPUT_SIZE; ++j) {
        sum += grads[(j + grads_0) * grads_1_shape + (0 + grads_1)];
    }
    neuronSum[(0 + neuronSum_0) * neuronSum_1_shape + (0 + neuronSum_1)] = sum;
}


void gradSumShell_gradSum(dacpp::Matrix<float> & matGrads, dacpp::Matrix<float> & matNeuronSum) {
    auto selector = default_selector_v;
    queue q(selector);
    ParameterGeneration para_gene_tool;
    
    DataInfo info_matGrads;
    info_matGrads.dim = matGrads.getDim();
    int info_matGrads_Shape[2] = {0};
    for(int i = 0; i < info_matGrads.dim; i++)
    {
        info_matGrads.dimLength.push_back(matGrads.getShape(i));
        info_matGrads_Shape[i] = matGrads.getShape(i);
    }
	
    DataInfo info_matNeuronSum;
    info_matNeuronSum.dim = matNeuronSum.getDim();
    int info_matNeuronSum_Shape[2] = {0};
    for(int i = 0; i < info_matNeuronSum.dim; i++)
    {
        info_matNeuronSum.dimLength.push_back(matNeuronSum.getShape(i));
        info_matNeuronSum_Shape[i] = matNeuronSum.getShape(i);
    }
	
    Index idx1 = Index("idx1");
    idx1.setDimId(1);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_matGrads));

    Index idx2 = Index("idx2");
    idx2.setDimId(1);
    idx2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx2,info_matNeuronSum));

	
	
    Dac_Ops matGrads_Ops;
    
    idx1.setDimId(1);
    matGrads_Ops.push_back(idx1);


    Dac_Ops matNeuronSum_Ops;
    
    idx1.setDimId(0);
    matNeuronSum_Ops.push_back(idx1);

    idx2.setDimId(1);
    matNeuronSum_Ops.push_back(idx2);


    Dac_Ops In_Ops;
    
    idx1.setDimId(0);
    In_Ops.push_back(idx1);

    idx2.setDimId(1);
    In_Ops.push_back(idx2);


    Dac_Ops Out_Ops;
    

	
	
	
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    
    
	
    float* h_matGrads = (float*)malloc(matGrads.getSize()*sizeof(float));
    matGrads.tensor2Array(h_matGrads);

    float* h_matNeuronSum = (float*)malloc(matNeuronSum.getSize()*sizeof(float));
    matNeuronSum.tensor2Array(h_matNeuronSum);

    
    Dac_Ops matGrads_ops;
    
    idx1.setDimId(1);
    matGrads_ops.push_back(idx1);


    auto r_matGrads = std::make_unique<sycl::buffer<float, 1>>(h_matGrads,sycl::range<1>(matGrads.getSize()));
    r_matGrads->set_final_data(h_matGrads);

	std::vector<int> info_partition_matGrads=para_gene_tool.init_partition_data_shape(info_matGrads,matGrads_ops);
    sycl::buffer<int> info_partition_matGrads_buffer(info_partition_matGrads.data(), sycl::range<1>(info_partition_matGrads.size()));

    
    Dac_Ops matNeuronSum_ops;
    
    idx1.setDimId(0);
    matNeuronSum_ops.push_back(idx1);
    idx2.setDimId(1);
    matNeuronSum_ops.push_back(idx2);


    auto r_matNeuronSum = std::make_unique<sycl::buffer<float, 1>>(h_matNeuronSum,sycl::range<1>(matNeuronSum.getSize()));
    r_matNeuronSum->set_final_data(h_matNeuronSum);

	std::vector<int> info_partition_matNeuronSum=para_gene_tool.init_partition_data_shape(info_matNeuronSum,matNeuronSum_ops);
    sycl::buffer<int> info_partition_matNeuronSum_buffer(info_partition_matNeuronSum.data(), sycl::range<1>(info_partition_matNeuronSum.size()));

	
	
    sycl::device device = q.get_device();
    auto max_sizes = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
    int max_global_size_x = max_sizes[0];
    int max_global_size_y = max_sizes[1];
    int max_global_size_z = max_sizes[2];
    int dim_x = (int)sycl::ceil(sycl::sqrt((float)Item_Size));
    int dim_y = (int)sycl::ceil((float)Item_Size / dim_x);
    int local_x = std::min(16, max_global_size_x);
    int local_y = std::min(16, max_global_size_y);
    int global_x = ((dim_x + local_x - 1) / local_x) * local_x;
    int global_y = ((dim_y + local_y - 1) / local_y) * local_y;

    sycl::range<2> local(local_x, local_y);
    sycl::range<2> global(global_x, global_y);
    q.submit([&](handler &h) {
    
        accessor<float, 1, sycl::access::mode::read_write> acc_matGrads(*r_matGrads, h);
        accessor<float, 1, sycl::access::mode::read_write> acc_matNeuronSum(*r_matNeuronSum, h);
    
        auto info_partition_matGrads_accessor = info_partition_matGrads_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_matNeuronSum_accessor = info_partition_matNeuronSum_buffer.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
			
            const auto idx1_=(item_id/idx2.split_size+(0))%idx1.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;
			
			const auto matGrads_0 = 0;
			const auto matGrads_1 = idx1_ * idx1.stride;
			const auto matNeuronSum_0 = idx1_ * idx1.stride;
			const auto matNeuronSum_1 = idx2_ * idx2.stride;
            
            auto* d_matGrads = acc_matGrads.get_multi_ptr<access::decorated::no>().get();
            auto* d_matNeuronSum = acc_matNeuronSum.get_multi_ptr<access::decorated::no>().get();
			
            gradSum(d_matGrads,d_matNeuronSum,matGrads_0,matGrads_1,matNeuronSum_0,matNeuronSum_1,info_matGrads_Shape[0],info_matGrads_Shape[1],info_matNeuronSum_Shape[0],info_matNeuronSum_Shape[1],info_partition_matGrads_accessor,info_partition_matNeuronSum_accessor);
        });
    }).wait();


	
    r_matGrads.reset();
    matGrads.array2Tensor(h_matGrads);

    r_matNeuronSum.reset();
    matNeuronSum.array2Tensor(h_matNeuronSum);

	

}

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
    gradSumShell_gradSum(matGrads, matNeuronSum);

    // Output results
    std::cout << "First 5 neuron gradient sums:\n";
    for (size_t i = 0; i < std::min(5, NUM_NEURONS) ; ++i)
        std::cout << matNeuronSum[i][0] << " ";
    std::cout << std::endl;

    return 0;
}