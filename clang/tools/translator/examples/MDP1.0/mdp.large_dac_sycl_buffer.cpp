#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <any>
#include <queue>
#include "ReconTensor.h"
namespace dacpp {
    typedef std::vector<std::any> list;
}
// Parameter settings
const double A = 1.0;  // Attraction coefficient
const double D = 0.1;  // Diffusion coefficient
const double dx = 0.1; // Spatial step size
const double dt = 0.01; // Time step size
const int N = 10000;     // Number of spatial grid points
const int T = 1000;    // Number of time steps


// Initialize user preference distribution
void initialize(std::vector<double>& p) {
    for (int i = 0; i < N; ++i) {
        // Assume initial preference follows a Gaussian distribution
        double x = i * dx;
        p[i] = std::exp(-std::pow(x - 5.0, 2) / 2.0); // Initial preference distribution centered at x=5
    }
}

// Normalization function
void normalize(dacpp::Vector<double>& p) {
    double sum = 0.0;
    for (int i = 0;i < N-2; i++) {
        sum += p[i];
    }
    for (int i = 0;i < N-2; i++) {
        p[i] /= sum; // Normalize
    }
}





// Numerically solve the Fokker-Planck equation

#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void mdp(double* p,double* new_p,int p_0,int new_p_0,int p_0_shape,int new_p_0_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_p_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_new_p_acc) {
    double diffusion = D * (p[2+p_0] - 2 * p[1+p_0] + p[0+p_0]) / (dx * dx);
    double drift = (-A) * (p[2+p_0] - p[0+p_0]) / (2 * dx);
    new_p[0+new_p_0] = p[1+p_0] + dt * (diffusion + drift);
}


void mdp_shell_mdp(dacpp::Vector<double> & p, dacpp::Vector<double> & new_p) {
    auto selector = default_selector_v;
    queue q(selector);
    ParameterGeneration para_gene_tool;
    
    DataInfo info_p;
    info_p.dim = p.getDim();
    int info_p_Shape[1] = {0};
    for(int i = 0; i < info_p.dim; i++)
    {
        info_p.dimLength.push_back(p.getShape(i));
        info_p_Shape[i] = p.getShape(i);
    }
	
    DataInfo info_new_p;
    info_new_p.dim = new_p.getDim();
    int info_new_p_Shape[1] = {0};
    for(int i = 0; i < info_new_p.dim; i++)
    {
        info_new_p.dimLength.push_back(new_p.getShape(i));
        info_new_p_Shape[i] = new_p.getShape(i);
    }
	
    RegularSlice sp = RegularSlice("sp", 3, 1);
    sp.setDimId(0);
    sp.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sp,info_p));

    Index idx = Index("idx");
    idx.setDimId(0);
    idx.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx,info_new_p));

	
	
    Dac_Ops p_Ops;
    
    sp.setDimId(0);
    p_Ops.push_back(sp);


    Dac_Ops new_p_Ops;
    
    idx.setDimId(0);
    new_p_Ops.push_back(idx);


    Dac_Ops In_Ops;
    
    sp.setDimId(0);
    In_Ops.push_back(sp);


    Dac_Ops Out_Ops;
    

	
	
	
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    
    
	
    double* h_p = (double*)malloc(p.getSize()*sizeof(double));
    p.tensor2Array(h_p);

    double* h_new_p = (double*)malloc(new_p.getSize()*sizeof(double));
    new_p.tensor2Array(h_new_p);

    
    Dac_Ops p_ops;
    
    sp.setDimId(0);
    p_ops.push_back(sp);


    auto r_p = std::make_unique<sycl::buffer<double, 1>>(h_p,sycl::range<1>(p.getSize()));
    r_p->set_final_data(h_p);

	std::vector<int> info_partition_p=para_gene_tool.init_partition_data_shape(info_p,p_ops);
    sycl::buffer<int> info_partition_p_buffer(info_partition_p.data(), sycl::range<1>(info_partition_p.size()));

    
    Dac_Ops new_p_ops;
    
    idx.setDimId(0);
    new_p_ops.push_back(idx);


    auto r_new_p = std::make_unique<sycl::buffer<double, 1>>(h_new_p,sycl::range<1>(new_p.getSize()));
    r_new_p->set_final_data(h_new_p);

	std::vector<int> info_partition_new_p=para_gene_tool.init_partition_data_shape(info_new_p,new_p_ops);
    sycl::buffer<int> info_partition_new_p_buffer(info_partition_new_p.data(), sycl::range<1>(info_partition_new_p.size()));

	
	
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

    for (int t = 0; t < T; ++t) {
        
    q.submit([&](handler &h) {
    
        accessor<double, 1, sycl::access::mode::read_write> acc_p(*r_p, h);
        accessor<double, 1, sycl::access::mode::read_write> acc_new_p(*r_new_p, h);
    
        auto info_partition_p_accessor = info_partition_p_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_new_p_accessor = info_partition_new_p_buffer.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
			
            const auto idx_=(item_id+(0))%idx.split_size;
            const auto sp_=(item_id+(0))%sp.split_size;
			
			const auto p_0 = sp_ * sp.stride;
			const auto new_p_0 = idx_ * idx.stride;
            
            auto* d_p = acc_p.get_multi_ptr<access::decorated::no>().get();
            auto* d_new_p = acc_new_p.get_multi_ptr<access::decorated::no>().get();
			
            mdp(d_p,d_new_p,p_0,new_p_0,info_p_Shape[0],info_new_p_Shape[0],info_partition_p_accessor,info_partition_new_p_accessor);
        });
    }).wait();

;
        //normalize(new_p); // Normalize distribution
        // Update distribution
        {
  int __L = (0);
  int __R = (N-3);
  int __N = __R - __L + 1;
  q.submit([&](sycl::handler& h){
    auto acc_p = r_p->get_access<sycl::access::mode::read_write>(h);
    auto acc_new_p = r_new_p->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>(__N), [=](sycl::id<1> idx){
      auto* d_p = acc_p.template get_multi_ptr<sycl::access::decorated::no>().get();
      auto* d_new_p = acc_new_p.template get_multi_ptr<sycl::access::decorated::no>().get();
      int i = __L + idx[0];
      {
            d_p[i+1] = d_new_p[i];
        }
    });
  });
}

        // Set boundary conditions
        //p[0] = 0.0;
        //p[N - 1] = 0.0;
        
    }


	
    r_p.reset();
    p.array2Tensor(h_p);

    r_new_p.reset();
    new_p.array2Tensor(h_new_p);

	

}

int main() {
    std::vector<double> p1(N, 0.0); // Store user preference distribution
    // Initialize preference distribution
    initialize(p1);
    // Numerically solve the Fokker-Planck equation
    std::vector<double> new_p1(N-2, 0.0); // Store distribution at next time step
    dacpp::Vector<double> p(p1);
    dacpp::Vector<double> new_p(new_p1);
    mdp_shell_mdp(p, new_p);
    
    std::cout << p[2] << std::endl;
    //p.print();
    return 0;
}
