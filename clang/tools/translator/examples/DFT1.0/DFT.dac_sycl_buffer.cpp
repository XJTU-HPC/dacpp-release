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







// Discrete Fourier Transform (DFT)
#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void dft(const std::complex<double>* input,std::complex<double>* output,const int* vec,int input_0,int output_0,int vec_0,int input_0_shape,int output_0_shape,int vec_0_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_input_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_output_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_vec_acc) {
    Complex sum(0, 0);
    for (int n = 0; n < N; ++n) {
        double angle = -2. * 3.1415926535897931 * vec[0+vec_0] * n / N;
        Complex W_n(std::cos(angle), std::sin(angle));
        sum += input[n+input_0] * W_n;
    }
    output[0+output_0] = sum;
}


void DFT_dft(const dacpp::Vector<std::complex<double> > & input, dacpp::Vector<std::complex<double> > & output, const dacpp::Vector<int> & vec) {
    auto selector = default_selector_v;
    queue q(selector);
    ParameterGeneration para_gene_tool;
    
    DataInfo info_input;
    info_input.dim = input.getDim();
    int info_input_Shape[1] = {0};
    for(int i = 0; i < info_input.dim; i++)
    {
        info_input.dimLength.push_back(input.getShape(i));
        info_input_Shape[i] = input.getShape(i);
    }
	
    DataInfo info_output;
    info_output.dim = output.getDim();
    int info_output_Shape[1] = {0};
    for(int i = 0; i < info_output.dim; i++)
    {
        info_output.dimLength.push_back(output.getShape(i));
        info_output_Shape[i] = output.getShape(i);
    }
	
    DataInfo info_vec;
    info_vec.dim = vec.getDim();
    int info_vec_Shape[1] = {0};
    for(int i = 0; i < info_vec.dim; i++)
    {
        info_vec.dimLength.push_back(vec.getShape(i));
        info_vec_Shape[i] = vec.getShape(i);
    }
	
    Index i = Index("i");
    i.setDimId(0);
    i.SetSplitSize(para_gene_tool.init_operetor_splitnumber(i,info_output));

	
	
    Dac_Ops input_Ops;
    

    Dac_Ops output_Ops;
    
    i.setDimId(0);
    output_Ops.push_back(i);


    Dac_Ops vec_Ops;
    
    i.setDimId(0);
    vec_Ops.push_back(i);


    Dac_Ops In_Ops;
    
    i.setDimId(0);
    In_Ops.push_back(i);


    Dac_Ops Out_Ops;
    
    i.setDimId(0);
    Out_Ops.push_back(i);


	
	
	
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    
    
	
    std::complex<double>* h_input = (std::complex<double>*)malloc(input.getSize()*sizeof(std::complex<double>));
    input.tensor2Array(h_input);
	buffer<std::complex<double>, 1> r_input(h_input, range<1>(input.getSize()));

    std::complex<double>* h_output = (std::complex<double>*)malloc(output.getSize()*sizeof(std::complex<double>));

    int* h_vec = (int*)malloc(vec.getSize()*sizeof(int));
    vec.tensor2Array(h_vec);
	buffer<int, 1> r_vec(h_vec, range<1>(vec.getSize()));


    
    Dac_Ops input_ops;
    


	std::vector<int> info_partition_input=para_gene_tool.init_partition_data_shape(info_input,input_ops);
    sycl::buffer<int> info_partition_input_buffer(info_partition_input.data(), sycl::range<1>(info_partition_input.size()));

    
    Dac_Ops output_ops;
    
    i.setDimId(0);
    output_ops.push_back(i);


    auto r_output = std::make_unique<sycl::buffer<std::complex<double>, 1>>(h_output,sycl::range<1>(output.getSize()));
    r_output->set_final_data(h_output);

	std::vector<int> info_partition_output=para_gene_tool.init_partition_data_shape(info_output,output_ops);
    sycl::buffer<int> info_partition_output_buffer(info_partition_output.data(), sycl::range<1>(info_partition_output.size()));


    
    Dac_Ops vec_ops;
    
    i.setDimId(0);
    vec_ops.push_back(i);


	std::vector<int> info_partition_vec=para_gene_tool.init_partition_data_shape(info_vec,vec_ops);
    sycl::buffer<int> info_partition_vec_buffer(info_partition_vec.data(), sycl::range<1>(info_partition_vec.size()));

	
	
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
    
        accessor<std::complex<double>, 1, access::mode::read> acc_input(r_input, h);
        r_input.set_final_data(nullptr);
        
        accessor<std::complex<double>, 1, sycl::access::mode::discard_write> acc_output(*r_output, h);
        accessor<int, 1, access::mode::read> acc_vec(r_vec, h);
        r_vec.set_final_data(nullptr);
        
    
        auto info_partition_input_accessor = info_partition_input_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_output_accessor = info_partition_output_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_vec_accessor = info_partition_vec_buffer.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
			
            const auto i_=(item_id+(0))%i.split_size;
			
			const auto input_0 = 0;
			const auto output_0 = i_ * i.stride;
			const auto vec_0 = i_ * i.stride;
            
            auto* d_input = acc_input.get_multi_ptr<access::decorated::no>().get();
            auto* d_output = acc_output.get_multi_ptr<access::decorated::no>().get();
            auto* d_vec = acc_vec.get_multi_ptr<access::decorated::no>().get();
			
            dft(d_input,d_output,d_vec,input_0,output_0,vec_0,info_input_Shape[0],info_output_Shape[0],info_vec_Shape[0],info_partition_input_accessor,info_partition_output_accessor,info_partition_vec_accessor);
        });
    }).wait();


	
    r_output.reset();
    output.array2Tensor(h_output);

	

}

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
    DFT_dft(input_tensor, output_tensor, vec_tensor);
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
    DFT_dft(input_tensor, output_tensor, vec_tensor);
    output_tensor.print();

    return 0;
}