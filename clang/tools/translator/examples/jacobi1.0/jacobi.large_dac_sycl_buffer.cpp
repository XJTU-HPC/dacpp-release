#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"
#define DACPP_TRANSLATE_MODE 1
// Define matrix size
const int N = 1000; // Modify N to change the matrix size
const int max_iter = 10000;
const float tolerance = 1e-6;
namespace dacpp {
    typedef std::vector<std::any> list;
}







#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void jacobi(const float* a,const float* b,const float* x,float* x_new,const int* num,int a_0,int a_1,int b_0,int x_0,int x_new_0,int num_0,int a_0_shape,int a_1_shape,int b_0_shape,int x_0_shape,int x_new_0_shape,int num_0_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_a_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_b_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_x_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_x_new_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_num_acc) {
    float sigma = 0;
    for (int i = 0; i < N; ++i) {
        if (i != num[0+num_0]) {
            sigma += a[(0 + a_0) * a_1_shape + (i + a_1)] * x[i+x_0];
        }
    }
    x_new[0+x_new_0] = (b[0+b_0] - sigma) / a[(0 + a_0) * a_1_shape + (num[0+num_0] + a_1)];
}


void jacobiShell_jacobi(const dacpp::Matrix<float> & A, const dacpp::Vector<float> & b, const dacpp::Vector<float> & x, dacpp::Vector<float> & x_new, const dacpp::Vector<int> & nums) {
    auto selector = default_selector_v;
    queue q(selector);
    ParameterGeneration para_gene_tool;
    
    DataInfo info_A;
    info_A.dim = A.getDim();
    int info_A_Shape[2] = {0};
    for(int i = 0; i < info_A.dim; i++)
    {
        info_A.dimLength.push_back(A.getShape(i));
        info_A_Shape[i] = A.getShape(i);
    }
	
    DataInfo info_b;
    info_b.dim = b.getDim();
    int info_b_Shape[1] = {0};
    for(int i = 0; i < info_b.dim; i++)
    {
        info_b.dimLength.push_back(b.getShape(i));
        info_b_Shape[i] = b.getShape(i);
    }
	
    DataInfo info_x;
    info_x.dim = x.getDim();
    int info_x_Shape[1] = {0};
    for(int i = 0; i < info_x.dim; i++)
    {
        info_x.dimLength.push_back(x.getShape(i));
        info_x_Shape[i] = x.getShape(i);
    }
	
    DataInfo info_x_new;
    info_x_new.dim = x_new.getDim();
    int info_x_new_Shape[1] = {0};
    for(int i = 0; i < info_x_new.dim; i++)
    {
        info_x_new.dimLength.push_back(x_new.getShape(i));
        info_x_new_Shape[i] = x_new.getShape(i);
    }
	
    DataInfo info_nums;
    info_nums.dim = nums.getDim();
    int info_nums_Shape[1] = {0};
    for(int i = 0; i < info_nums.dim; i++)
    {
        info_nums.dimLength.push_back(nums.getShape(i));
        info_nums_Shape[i] = nums.getShape(i);
    }
	
    Index idx1 = Index("idx1");
    idx1.setDimId(0);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_A));

	
	
    Dac_Ops A_Ops;
    
    idx1.setDimId(0);
    A_Ops.push_back(idx1);


    Dac_Ops b_Ops;
    
    idx1.setDimId(0);
    b_Ops.push_back(idx1);


    Dac_Ops x_Ops;
    

    Dac_Ops x_new_Ops;
    
    idx1.setDimId(0);
    x_new_Ops.push_back(idx1);


    Dac_Ops nums_Ops;
    
    idx1.setDimId(0);
    nums_Ops.push_back(idx1);


    Dac_Ops In_Ops;
    
    idx1.setDimId(0);
    In_Ops.push_back(idx1);


    Dac_Ops Out_Ops;
    
    idx1.setDimId(0);
    Out_Ops.push_back(idx1);


	
	
	
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    
    
	
    float* h_A = (float*)malloc(A.getSize()*sizeof(float));
    A.tensor2Array(h_A);
	buffer<float, 1> r_A(h_A, range<1>(A.getSize()));

    float* h_b = (float*)malloc(b.getSize()*sizeof(float));
    b.tensor2Array(h_b);
	buffer<float, 1> r_b(h_b, range<1>(b.getSize()));

    float* h_x = (float*)malloc(x.getSize()*sizeof(float));
    x.tensor2Array(h_x);
	buffer<float, 1> r_x(h_x, range<1>(x.getSize()));

    float* h_x_new = (float*)malloc(x_new.getSize()*sizeof(float));

    int* h_nums = (int*)malloc(nums.getSize()*sizeof(int));
    nums.tensor2Array(h_nums);
	buffer<int, 1> r_nums(h_nums, range<1>(nums.getSize()));


    
    Dac_Ops A_ops;
    
    idx1.setDimId(0);
    A_ops.push_back(idx1);


	std::vector<int> info_partition_A=para_gene_tool.init_partition_data_shape(info_A,A_ops);
    sycl::buffer<int> info_partition_A_buffer(info_partition_A.data(), sycl::range<1>(info_partition_A.size()));


    
    Dac_Ops b_ops;
    
    idx1.setDimId(0);
    b_ops.push_back(idx1);


	std::vector<int> info_partition_b=para_gene_tool.init_partition_data_shape(info_b,b_ops);
    sycl::buffer<int> info_partition_b_buffer(info_partition_b.data(), sycl::range<1>(info_partition_b.size()));


    
    Dac_Ops x_ops;
    


	std::vector<int> info_partition_x=para_gene_tool.init_partition_data_shape(info_x,x_ops);
    sycl::buffer<int> info_partition_x_buffer(info_partition_x.data(), sycl::range<1>(info_partition_x.size()));

    
    Dac_Ops x_new_ops;
    
    idx1.setDimId(0);
    x_new_ops.push_back(idx1);


    auto r_x_new = std::make_unique<sycl::buffer<float, 1>>(h_x_new,sycl::range<1>(x_new.getSize()));
    r_x_new->set_final_data(h_x_new);

	std::vector<int> info_partition_x_new=para_gene_tool.init_partition_data_shape(info_x_new,x_new_ops);
    sycl::buffer<int> info_partition_x_new_buffer(info_partition_x_new.data(), sycl::range<1>(info_partition_x_new.size()));


    
    Dac_Ops nums_ops;
    
    idx1.setDimId(0);
    nums_ops.push_back(idx1);


	std::vector<int> info_partition_nums=para_gene_tool.init_partition_data_shape(info_nums,nums_ops);
    sycl::buffer<int> info_partition_nums_buffer(info_partition_nums.data(), sycl::range<1>(info_partition_nums.size()));

	
	
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
    
        accessor<float, 1, access::mode::read> acc_A(r_A, h);
        r_A.set_final_data(nullptr);
        
        accessor<float, 1, access::mode::read> acc_b(r_b, h);
        r_b.set_final_data(nullptr);
        
        accessor<float, 1, access::mode::read> acc_x(r_x, h);
        r_x.set_final_data(nullptr);
        
        accessor<float, 1, sycl::access::mode::discard_write> acc_x_new(*r_x_new, h);
        accessor<int, 1, access::mode::read> acc_nums(r_nums, h);
        r_nums.set_final_data(nullptr);
        
    
        auto info_partition_A_accessor = info_partition_A_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_b_accessor = info_partition_b_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_x_accessor = info_partition_x_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_x_new_accessor = info_partition_x_new_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_nums_accessor = info_partition_nums_buffer.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
			
            const auto idx1_=(item_id+(0))%idx1.split_size;
			
			const auto A_0 = idx1_ * idx1.stride;
			const auto A_1 = 0;
			const auto b_0 = idx1_ * idx1.stride;
			const auto x_0 = 0;
			const auto x_new_0 = idx1_ * idx1.stride;
			const auto nums_0 = idx1_ * idx1.stride;
            
            auto* d_A = acc_A.get_multi_ptr<access::decorated::no>().get();
            auto* d_b = acc_b.get_multi_ptr<access::decorated::no>().get();
            auto* d_x = acc_x.get_multi_ptr<access::decorated::no>().get();
            auto* d_x_new = acc_x_new.get_multi_ptr<access::decorated::no>().get();
            auto* d_nums = acc_nums.get_multi_ptr<access::decorated::no>().get();
			
            jacobi(d_A,d_b,d_x,d_x_new,d_nums,A_0,A_1,b_0,x_0,x_new_0,nums_0,info_A_Shape[0],info_A_Shape[1],info_b_Shape[0],info_x_Shape[0],info_x_new_Shape[0],info_nums_Shape[0],info_partition_A_accessor,info_partition_b_accessor,info_partition_x_accessor,info_partition_x_new_accessor,info_partition_nums_accessor);
        });
    }).wait();


	
    r_x_new.reset();
    x_new.array2Tensor(h_x_new);

	

}

int main() {


    // Initialize coefficient matrix A and vector b
    std::vector<float> mat_A(N * N, 0.0f);
    std::vector<float> vec_b(N, 0.0f);
    std::vector<float> vec_x(N, 0.0f);     // Initial solution
    std::vector<float> vec_x_new(N, 0.0f); // Updated solution

    // Auto-initialize A and b
    for (int i = 0; i < N; ++i) {
        mat_A[i * N + i] = 4.0f; // Diagonal elements, ensure diagonal dominance

        if (i > 0) {
            mat_A[i * N + i - 1] = -1.0f; // Lower triangular elements
        }
        if (i < N - 1) {
            mat_A[i * N + i + 1] = -1.0f; // Upper triangular elements
        }

        vec_b[i] = 1.0f; // Initialize vector b, modify as needed
    }

    // std::vector<int> A_shape = {100, 100};
    // std::vector<int> b_shape = {100};
    // std::vector<int> x_shape = {100};
    // std::vector<int> x_new_shape = {100};
    dacpp::Matrix<float> A({N, N}, mat_A);
    dacpp::Vector<float> b(vec_b);
    dacpp::Vector<float> x(vec_x);
    dacpp::Vector<float> x_new(vec_x_new);
    
    bool converged = false;
    int iter = 0;
    std::vector<int> nums(N);
    // Fill nums using std::iota, starting from 0
    for(int i = 0;i < N;  i++){
        nums[i] = i;
    }
    //std::vector<int> nums_shape = {100};
    dacpp::Vector<int> tensor_nums(nums);
    float* data = new float[1 * N];
    float* data2 = new float[1 * N];

    while (!converged && iter < max_iter) {
        jacobiShell_jacobi(A, b, x, x_new, tensor_nums);
        
        x.tensor2Array(data);
        x_new.tensor2Array(data2);

        float max_error = 0.0f;
        for (int i = 0; i < N; ++i) {
            max_error = std::max(max_error, std::fabs(data2[i] - data[i]));
        }

        if (max_error < tolerance) {
            converged = true;
        }

        // Update x
        x=x_new;

        ++iter;
    }


    // Output results
    //std::cout << "Iterations: " << iter << std::endl;
    //std::cout << "Solution vector x:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << data2[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
