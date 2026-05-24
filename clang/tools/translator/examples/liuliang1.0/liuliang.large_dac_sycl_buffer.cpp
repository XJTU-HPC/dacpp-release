#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <any>
#include <queue>
#include "ReconTensor.h"


namespace dacpp {
    typedef std::vector<std::any> list;
}

const int WIDTH = 250;       // Road segment length
const double TIME_STEPS = 500;  // Number of time steps
const double DELTA_T = 0.01; // Time step size
const double DELTA_X = 1.0;  // Spatial step size

// Flow function, considering the effect of density on flow
double q(double rho) {
    double V_max = 30; // Maximum velocity
    double rho_max = 50; // Maximum density
    return rho * V_max * (1 - rho / rho_max);
}

// Initialize density using a random distribution
void initializeDensity(std::vector<double>& rho) {
    for (int i = 0; i < WIDTH; ++i) {
        if (i < WIDTH / 4) {
            rho[i] = 40; // High density region
        } else if (i < 3 * WIDTH / 4) {
            rho[i] = 20; // Medium density region
        } else {
            rho[i] = 10; // Low density region
        }
    }
}

// Calculate traffic flow




#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void lwr(double* rho,double* new_rho,int rho_0,int new_rho_0,int rho_0_shape,int new_rho_0_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_rho_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_new_rho_acc) {
    new_rho[0+new_rho_0] = rho[1+rho_0] - (DELTA_T / DELTA_X) * (q(rho[1+rho_0]) - q(rho[0+rho_0]));
    new_rho[0+new_rho_0] = std::max(0., new_rho[0+new_rho_0]);
}


void LWR_shell_lwr(dacpp::Vector<double> & rho, dacpp::Vector<double> & new_rho) {
    auto selector = default_selector_v;
    queue q(selector);
    ParameterGeneration para_gene_tool;
    
    DataInfo info_rho;
    info_rho.dim = rho.getDim();
    int info_rho_Shape[1] = {0};
    for(int i = 0; i < info_rho.dim; i++)
    {
        info_rho.dimLength.push_back(rho.getShape(i));
        info_rho_Shape[i] = rho.getShape(i);
    }
	
    DataInfo info_new_rho;
    info_new_rho.dim = new_rho.getDim();
    int info_new_rho_Shape[1] = {0};
    for(int i = 0; i < info_new_rho.dim; i++)
    {
        info_new_rho.dimLength.push_back(new_rho.getShape(i));
        info_new_rho_Shape[i] = new_rho.getShape(i);
    }
	
    RegularSlice S1 = RegularSlice("S1", 2, 1);
    S1.setDimId(0);
    S1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(S1,info_rho));

    Index idx1 = Index("idx1");
    idx1.setDimId(0);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_new_rho));

	
	
    Dac_Ops rho_Ops;
    
    S1.setDimId(0);
    rho_Ops.push_back(S1);


    Dac_Ops new_rho_Ops;
    
    idx1.setDimId(0);
    new_rho_Ops.push_back(idx1);


    Dac_Ops In_Ops;
    
    S1.setDimId(0);
    In_Ops.push_back(S1);


    Dac_Ops Out_Ops;
    

	
	
	
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    
    
	
    double* h_rho = (double*)malloc(rho.getSize()*sizeof(double));
    rho.tensor2Array(h_rho);

    double* h_new_rho = (double*)malloc(new_rho.getSize()*sizeof(double));
    new_rho.tensor2Array(h_new_rho);

    
    Dac_Ops rho_ops;
    
    S1.setDimId(0);
    rho_ops.push_back(S1);


    auto r_rho = std::make_unique<sycl::buffer<double, 1>>(h_rho,sycl::range<1>(rho.getSize()));
    r_rho->set_final_data(h_rho);

	std::vector<int> info_partition_rho=para_gene_tool.init_partition_data_shape(info_rho,rho_ops);
    sycl::buffer<int> info_partition_rho_buffer(info_partition_rho.data(), sycl::range<1>(info_partition_rho.size()));

    
    Dac_Ops new_rho_ops;
    
    idx1.setDimId(0);
    new_rho_ops.push_back(idx1);


    auto r_new_rho = std::make_unique<sycl::buffer<double, 1>>(h_new_rho,sycl::range<1>(new_rho.getSize()));
    r_new_rho->set_final_data(h_new_rho);

	std::vector<int> info_partition_new_rho=para_gene_tool.init_partition_data_shape(info_new_rho,new_rho_ops);
    sycl::buffer<int> info_partition_new_rho_buffer(info_partition_new_rho.data(), sycl::range<1>(info_partition_new_rho.size()));

	
	
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

    for (int t = 0; t < TIME_STEPS; ++t) {
        
    q.submit([&](handler &h) {
    
        accessor<double, 1, sycl::access::mode::read_write> acc_rho(*r_rho, h);
        accessor<double, 1, sycl::access::mode::read_write> acc_new_rho(*r_new_rho, h);
    
        auto info_partition_rho_accessor = info_partition_rho_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_new_rho_accessor = info_partition_new_rho_buffer.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
			
            const auto idx1_=(item_id+(0))%idx1.split_size;
            const auto S1_=(item_id+(0))%S1.split_size;
			
			const auto rho_0 = S1_ * S1.stride;
			const auto new_rho_0 = idx1_ * idx1.stride;
            
            auto* d_rho = acc_rho.get_multi_ptr<access::decorated::no>().get();
            auto* d_new_rho = acc_new_rho.get_multi_ptr<access::decorated::no>().get();
			
            lwr(d_rho,d_new_rho,rho_0,new_rho_0,info_rho_Shape[0],info_new_rho_Shape[0],info_partition_rho_accessor,info_partition_new_rho_accessor);
        });
    }).wait();

;
        {
  int __L = (1);
  int __R = (WIDTH-2);
  int __N = __R - __L + 1;
  q.submit([&](sycl::handler& h){
    auto acc_rho = r_rho->get_access<sycl::access::mode::read_write>(h);
    auto acc_new_rho = r_new_rho->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>(__N), [=](sycl::id<1> idx){
      auto* d_rho = acc_rho.template get_multi_ptr<sycl::access::decorated::no>().get();
      auto* d_new_rho = acc_new_rho.template get_multi_ptr<sycl::access::decorated::no>().get();
      int i = __L + idx[0];
      {
            d_rho[i] = d_new_rho[i-1];
        }
    });
  });
}

        {
  int __L = (0);
  int __R = (1);
  int __N = __R - __L;
  q.submit([&](sycl::handler& h){
    auto acc_rho = r_rho->get_access<sycl::access::mode::read_write>(h);
    auto acc_new_rho = r_new_rho->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>(__N), [=](sycl::id<1> idx){
      auto* d_rho = acc_rho.template get_multi_ptr<sycl::access::decorated::no>().get();
      auto* d_new_rho = acc_new_rho.template get_multi_ptr<sycl::access::decorated::no>().get();
      int i = __L + idx[0];
      {
        d_rho[0] = d_new_rho[0]; // Left boundary: no traffic flow
        //middle_in_tensor[99] = middle_out_tensor[97];
    }
    });
  });
}

    }


	
    r_rho.reset();
    rho.array2Tensor(h_rho);

    r_new_rho.reset();
    new_rho.array2Tensor(h_new_rho);

	

}

int main() {
    // Create Tensor type objects
    std::vector<double> rho1(WIDTH, 0.0);
    std::vector<double> new_rho1(WIDTH, 0.0);
    initializeDensity(rho1);
    dacpp::Vector<double> rho_tensor(rho1);
    dacpp::Vector<double> new_rho_tensor(new_rho1);
    dacpp::Vector<double> new_rho = new_rho_tensor[{1,WIDTH-1}];
    dacpp::Vector<double> rho = rho_tensor[{0,WIDTH-1}];
    LWR_shell_lwr(rho, new_rho);
    
    std::cout << rho[15] << std::endl;
    

    

    // Free dynamically allocated memory

    return 0;
}
