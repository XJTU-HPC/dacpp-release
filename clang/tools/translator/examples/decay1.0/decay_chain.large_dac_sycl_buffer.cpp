#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "ReconTensor.h"
namespace dacpp {
    typedef std::vector<std::any> list;
}


const double dt = 0.1;       // Time step size
const double T = 5.0;       // Total time
const size_t numIsotopes = 10000; // Set a large number of isotopes (e.g., 10000)




// Calculate the quantity of each isotope at time t
#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void decay(const double* N0s,const double* lambdas,double* local_A,const double* t,int N0s_0,int lambdas_0,int local_A_0,int t_0,int N0s_0_shape,int lambdas_0_shape,int local_A_0_shape,int t_0_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_N0s_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_lambdas_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_local_A_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_t_acc) {
    local_A[0+local_A_0] = N0s[0+N0s_0] * std::exp(-lambdas[0+lambdas_0] * t[0+t_0]);
}


void DECAY_decay(const dacpp::Vector<double> & N0s, const dacpp::Vector<double> & lambdas, dacpp::Vector<double> & local_A, const dacpp::Vector<double> & t) {
    auto selector = default_selector_v;
    queue q(selector);
    ParameterGeneration para_gene_tool;
    
    DataInfo info_N0s;
    info_N0s.dim = N0s.getDim();
    int info_N0s_Shape[1] = {0};
    for(int i = 0; i < info_N0s.dim; i++)
    {
        info_N0s.dimLength.push_back(N0s.getShape(i));
        info_N0s_Shape[i] = N0s.getShape(i);
    }
	
    DataInfo info_lambdas;
    info_lambdas.dim = lambdas.getDim();
    int info_lambdas_Shape[1] = {0};
    for(int i = 0; i < info_lambdas.dim; i++)
    {
        info_lambdas.dimLength.push_back(lambdas.getShape(i));
        info_lambdas_Shape[i] = lambdas.getShape(i);
    }
	
    DataInfo info_local_A;
    info_local_A.dim = local_A.getDim();
    int info_local_A_Shape[1] = {0};
    for(int i = 0; i < info_local_A.dim; i++)
    {
        info_local_A.dimLength.push_back(local_A.getShape(i));
        info_local_A_Shape[i] = local_A.getShape(i);
    }
	
    DataInfo info_t;
    info_t.dim = t.getDim();
    int info_t_Shape[1] = {0};
    for(int i = 0; i < info_t.dim; i++)
    {
        info_t.dimLength.push_back(t.getShape(i));
        info_t_Shape[i] = t.getShape(i);
    }
	
    Index i = Index("i");
    i.setDimId(0);
    i.SetSplitSize(para_gene_tool.init_operetor_splitnumber(i,info_N0s));

	
	
    Dac_Ops N0s_Ops;
    
    i.setDimId(0);
    N0s_Ops.push_back(i);


    Dac_Ops lambdas_Ops;
    
    i.setDimId(0);
    lambdas_Ops.push_back(i);


    Dac_Ops local_A_Ops;
    
    i.setDimId(0);
    local_A_Ops.push_back(i);


    Dac_Ops t_Ops;
    

    Dac_Ops In_Ops;
    
    i.setDimId(0);
    In_Ops.push_back(i);


    Dac_Ops Out_Ops;
    
    i.setDimId(0);
    Out_Ops.push_back(i);


	
	
	
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    
    
	
    double* h_N0s = (double*)malloc(N0s.getSize()*sizeof(double));
    N0s.tensor2Array(h_N0s);
	buffer<double, 1> r_N0s(h_N0s, range<1>(N0s.getSize()));

    double* h_lambdas = (double*)malloc(lambdas.getSize()*sizeof(double));
    lambdas.tensor2Array(h_lambdas);
	buffer<double, 1> r_lambdas(h_lambdas, range<1>(lambdas.getSize()));

    double* h_local_A = (double*)malloc(local_A.getSize()*sizeof(double));

    double* h_t = (double*)malloc(t.getSize()*sizeof(double));
    t.tensor2Array(h_t);
	buffer<double, 1> r_t(h_t, range<1>(t.getSize()));


    
    Dac_Ops N0s_ops;
    
    i.setDimId(0);
    N0s_ops.push_back(i);


	std::vector<int> info_partition_N0s=para_gene_tool.init_partition_data_shape(info_N0s,N0s_ops);
    sycl::buffer<int> info_partition_N0s_buffer(info_partition_N0s.data(), sycl::range<1>(info_partition_N0s.size()));


    
    Dac_Ops lambdas_ops;
    
    i.setDimId(0);
    lambdas_ops.push_back(i);


	std::vector<int> info_partition_lambdas=para_gene_tool.init_partition_data_shape(info_lambdas,lambdas_ops);
    sycl::buffer<int> info_partition_lambdas_buffer(info_partition_lambdas.data(), sycl::range<1>(info_partition_lambdas.size()));

    
    Dac_Ops local_A_ops;
    
    i.setDimId(0);
    local_A_ops.push_back(i);


    auto r_local_A = std::make_unique<sycl::buffer<double, 1>>(h_local_A,sycl::range<1>(local_A.getSize()));
    r_local_A->set_final_data(h_local_A);

	std::vector<int> info_partition_local_A=para_gene_tool.init_partition_data_shape(info_local_A,local_A_ops);
    sycl::buffer<int> info_partition_local_A_buffer(info_partition_local_A.data(), sycl::range<1>(info_partition_local_A.size()));


    
    Dac_Ops t_ops;
    


	std::vector<int> info_partition_t=para_gene_tool.init_partition_data_shape(info_t,t_ops);
    sycl::buffer<int> info_partition_t_buffer(info_partition_t.data(), sycl::range<1>(info_partition_t.size()));

	
	
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
    
        accessor<double, 1, access::mode::read> acc_N0s(r_N0s, h);
        r_N0s.set_final_data(nullptr);
        
        accessor<double, 1, access::mode::read> acc_lambdas(r_lambdas, h);
        r_lambdas.set_final_data(nullptr);
        
        accessor<double, 1, sycl::access::mode::discard_write> acc_local_A(*r_local_A, h);
        accessor<double, 1, access::mode::read> acc_t(r_t, h);
        r_t.set_final_data(nullptr);
        
    
        auto info_partition_N0s_accessor = info_partition_N0s_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_lambdas_accessor = info_partition_lambdas_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_local_A_accessor = info_partition_local_A_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_t_accessor = info_partition_t_buffer.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
			
            const auto i_=(item_id+(0))%i.split_size;
			
			const auto N0s_0 = i_ * i.stride;
			const auto lambdas_0 = i_ * i.stride;
			const auto local_A_0 = i_ * i.stride;
			const auto t_0 = 0;
            
            auto* d_N0s = acc_N0s.get_multi_ptr<access::decorated::no>().get();
            auto* d_lambdas = acc_lambdas.get_multi_ptr<access::decorated::no>().get();
            auto* d_local_A = acc_local_A.get_multi_ptr<access::decorated::no>().get();
            auto* d_t = acc_t.get_multi_ptr<access::decorated::no>().get();
			
            decay(d_N0s,d_lambdas,d_local_A,d_t,N0s_0,lambdas_0,local_A_0,t_0,info_N0s_Shape[0],info_lambdas_Shape[0],info_local_A_Shape[0],info_t_Shape[0],info_partition_N0s_accessor,info_partition_lambdas_accessor,info_partition_local_A_accessor,info_partition_t_accessor);
        });
    }).wait();


	
    r_local_A.reset();
    local_A.array2Tensor(h_local_A);

	

}

void calculateDecay(const std::vector<double>& lambdas, const std::vector<double>& N0s, double dt, double T) {
    size_t numIsotopes = lambdas.size(); // Number of isotopes
    std::vector<double> A(T/dt*numIsotopes, 0.0);  // Store the quantity of each isotope at different time points
    std::vector<double> time;  // Time series
    std::vector<double> t;
    t.push_back(static_cast<double>(0));

    // Serial computation of the decay process for each isotope
    std::vector<double> local_A(numIsotopes, 0.0);
    dacpp::Vector<double> local_A_tensor(local_A);
    dacpp::Vector<double> N0s_tensor(N0s);
    dacpp::Vector<double> lambdas_tensor(lambdas);
    dacpp::Vector<double> t_tensor(t);
    dacpp::Matrix<double> A_tensor({static_cast<int>(T/dt), static_cast<int>(numIsotopes)}, A);
    

    while(t_tensor[0] <= T){  
        DECAY_decay(N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor);
        A_tensor[10*t_tensor[0]] = local_A_tensor;
        t_tensor[0] += dt;
    }
    A_tensor[1].print();
}

int main() {
    

    // Randomly generate decay constants and initial quantities
    std::vector<double> lambdas(numIsotopes);
    std::vector<double> N0s(numIsotopes, 1000.0);  // Initial quantity is 1000

    // Randomly initialize decay constants (e.g., lambda between 0.01 and 0.2)
    for (size_t i = 0; i < numIsotopes; ++i) {
        lambdas[i] = 0.01 + 0.01*i;  // lambda range [0.01, 0.2]
    }

    calculateDecay(lambdas, N0s, dt, T);

    return 0;
}