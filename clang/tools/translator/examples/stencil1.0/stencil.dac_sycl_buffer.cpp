#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"
// #define DACPP_TRANSLATE_MODE 1

using namespace std;
namespace dacpp {
    typedef std::vector<std::any> list;
}


// Grid parameters
const int NX = 8;           // Number of grid points in x direction
const int NY = 8;           // Number of grid points in y direction
const double Lx = 10.0f;       // Length in x direction
const double Ly = 10.0f;       // Length in y direction
const double alpha = 0.01f;    // Thermal diffusion coefficient
const int TIME_STEPS = 10;  // Number of time steps
// Spatial step size
const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);

// Stability condition
const double dt_stability = (dx * dx * dy * dy) / (2.0f * alpha * (dx * dx + dy * dy));
const double delta_t = 0.4f * dt_stability; // Choose a more restrictive time step to ensure stability





#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void stencil(double* mat,double* out,int mat_0,int mat_1,int out_0,int out_1,int mat_0_shape,int mat_1_shape,int out_0_shape,int out_1_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_mat_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_out_acc) {
    out[(0 + out_0) * out_1_shape + (0 + out_1)] = mat[(1 + mat_0) * mat_1_shape + (1 + mat_1)] + alpha * delta_t * (((mat[(2 + mat_0) * mat_1_shape + (1 + mat_1)] - 2.F * mat[(1 + mat_0) * mat_1_shape + (1 + mat_1)] + mat[(0 + mat_0) * mat_1_shape + (1 + mat_1)]) / (dx * dx)) + ((mat[(1 + mat_0) * mat_1_shape + (2 + mat_1)] - 2.F * mat[(1 + mat_0) * mat_1_shape + (1 + mat_1)] + mat[(1 + mat_0) * mat_1_shape + (0 + mat_1)]) / (dy * dy)));
}


void stencilShell_stencil(dacpp::Matrix<double> & matIn, dacpp::Matrix<double> & matOut) {
    auto selector = default_selector_v;
    queue q(selector);
    ParameterGeneration para_gene_tool;
    
    DataInfo info_matIn;
    info_matIn.dim = matIn.getDim();
    int info_matIn_Shape[2] = {0};
    for(int i = 0; i < info_matIn.dim; i++)
    {
        info_matIn.dimLength.push_back(matIn.getShape(i));
        info_matIn_Shape[i] = matIn.getShape(i);
    }
	
    DataInfo info_matOut;
    info_matOut.dim = matOut.getDim();
    int info_matOut_Shape[2] = {0};
    for(int i = 0; i < info_matOut.dim; i++)
    {
        info_matOut.dimLength.push_back(matOut.getShape(i));
        info_matOut_Shape[i] = matOut.getShape(i);
    }
	
    RegularSlice sp1 = RegularSlice("sp1", 3, 1);
    sp1.setDimId(0);
    sp1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sp1,info_matIn));

    RegularSlice sp2 = RegularSlice("sp2", 3, 1);
    sp2.setDimId(1);
    sp2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sp2,info_matIn));

    Index idx1 = Index("idx1");
    idx1.setDimId(0);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_matOut));

    Index idx2 = Index("idx2");
    idx2.setDimId(1);
    idx2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx2,info_matOut));

	
	
    Dac_Ops matIn_Ops;
    
    sp1.setDimId(0);
    matIn_Ops.push_back(sp1);

    sp2.setDimId(1);
    matIn_Ops.push_back(sp2);


    Dac_Ops matOut_Ops;
    
    idx1.setDimId(0);
    matOut_Ops.push_back(idx1);

    idx2.setDimId(1);
    matOut_Ops.push_back(idx2);


    Dac_Ops In_Ops;
    
    sp1.setDimId(0);
    In_Ops.push_back(sp1);

    sp2.setDimId(1);
    In_Ops.push_back(sp2);


    Dac_Ops Out_Ops;
    

	
	
	
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    
    
	
    double* h_matIn = (double*)malloc(matIn.getSize()*sizeof(double));
    matIn.tensor2Array(h_matIn);

    double* h_matOut = (double*)malloc(matOut.getSize()*sizeof(double));
    matOut.tensor2Array(h_matOut);

    
    Dac_Ops matIn_ops;
    
    sp1.setDimId(0);
    matIn_ops.push_back(sp1);
    sp2.setDimId(1);
    matIn_ops.push_back(sp2);


    auto r_matIn = std::make_unique<sycl::buffer<double, 1>>(h_matIn,sycl::range<1>(matIn.getSize()));
    r_matIn->set_final_data(h_matIn);

	std::vector<int> info_partition_matIn=para_gene_tool.init_partition_data_shape(info_matIn,matIn_ops);
    sycl::buffer<int> info_partition_matIn_buffer(info_partition_matIn.data(), sycl::range<1>(info_partition_matIn.size()));

    
    Dac_Ops matOut_ops;
    
    idx1.setDimId(0);
    matOut_ops.push_back(idx1);
    idx2.setDimId(1);
    matOut_ops.push_back(idx2);


    auto r_matOut = std::make_unique<sycl::buffer<double, 1>>(h_matOut,sycl::range<1>(matOut.getSize()));
    r_matOut->set_final_data(h_matOut);

	std::vector<int> info_partition_matOut=para_gene_tool.init_partition_data_shape(info_matOut,matOut_ops);
    sycl::buffer<int> info_partition_matOut_buffer(info_partition_matOut.data(), sycl::range<1>(info_partition_matOut.size()));

	
	
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

    for(int i=0;i<TIME_STEPS;i++) {
        
    q.submit([&](handler &h) {
    
        accessor<double, 1, sycl::access::mode::read_write> acc_matIn(*r_matIn, h);
        accessor<double, 1, sycl::access::mode::read_write> acc_matOut(*r_matOut, h);
    
        auto info_partition_matIn_accessor = info_partition_matIn_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_matOut_accessor = info_partition_matOut_buffer.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
			
            const auto sp1_=(item_id/sp2.split_size+(0))%sp1.split_size;
            const auto idx1_=(item_id/sp2.split_size+(0))%idx1.split_size;
            const auto sp2_=(item_id+(0))%sp2.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;
			
			const auto matIn_0 = sp1_ * sp1.stride;
			const auto matIn_1 = sp2_ * sp2.stride;
			const auto matOut_0 = idx1_ * idx1.stride;
			const auto matOut_1 = idx2_ * idx2.stride;
            
            auto* d_matIn = acc_matIn.get_multi_ptr<access::decorated::no>().get();
            auto* d_matOut = acc_matOut.get_multi_ptr<access::decorated::no>().get();
			
            stencil(d_matIn,d_matOut,matIn_0,matIn_1,matOut_0,matOut_1,info_matIn_Shape[0],info_matIn_Shape[1],info_matOut_Shape[0],info_matOut_Shape[1],info_partition_matIn_accessor,info_partition_matOut_accessor);
        });
    }).wait();

;

        {
    int __iL = (1);
    int __iR = (NX-2);
    int __iN = __iR - __iL + 1;
    int __jL = (1);
    int __jR = (NY-2);
    int __jN = __jR - __jL + 1;
    q.submit([&](sycl::handler& h){
    auto acc_matIn = r_matIn->get_access<sycl::access::mode::read_write>(h);
    auto acc_matOut = r_matOut->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<2>(__iN, __jN), [=](sycl::id<2> idx){
      auto* d_matIn = acc_matIn.template get_multi_ptr<sycl::access::decorated::no>().get();
      auto* d_matOut = acc_matOut.template get_multi_ptr<sycl::access::decorated::no>().get();
      int i = __iL + idx[0];
      int j = __jL + idx[1];
      {
                d_matIn[(i) * info_matIn_Shape[1] + (j)]=d_matOut[(i-1) * info_matOut_Shape[1] + (j-1)];
            }
    });
  });
}


        // Handle boundary conditions (adiabatic boundary: zero derivative)
        {
  int __L = (0);
  int __R = (NY-1);
  int __N = __R - __L + 1;
  q.submit([&](sycl::handler& h){
    auto acc_matIn = r_matIn->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>(__N), [=](sycl::id<1> idx){
      auto* d_matIn = acc_matIn.template get_multi_ptr<sycl::access::decorated::no>().get();
      int j = __L + idx[0];
      {
            //double* data = new double[1];
            d_matIn[(0) * info_matIn_Shape[1] + (j)]=d_matIn[(1) * info_matIn_Shape[1] + (j)];              // Top boundary
            d_matIn[(NX - 1) * info_matIn_Shape[1] + (j)]=d_matIn[(NX-2) * info_matIn_Shape[1] + (j)];
             // Bottom boundary
        }
    });
  });
}

        {
  int __L = (0);
  int __R = (NX-1);
  int __N = __R - __L;
  q.submit([&](sycl::handler& h){
    auto acc_matIn = r_matIn->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>(__N), [=](sycl::id<1> idx){
      auto* d_matIn = acc_matIn.template get_multi_ptr<sycl::access::decorated::no>().get();
      int i = __L + idx[0];
      {
            //double* data = new double[1];
            d_matIn[(i) * info_matIn_Shape[1] + (0)]=d_matIn[(i) * info_matIn_Shape[1] + (1)];              // Left boundary
            d_matIn[(i) * info_matIn_Shape[1] + (NY-1)]=d_matIn[(i) * info_matIn_Shape[1] + (NY-2)];
        }
    });
  });
}


    }


	
    r_matIn.reset();
    matIn.array2Tensor(h_matIn);

    r_matOut.reset();
    matOut.array2Tensor(h_matOut);

	

}

int main() {

    // Initialize temperature field
    vector<double> u_prev(NX * NY, 0.0f); // Previous step (in the heat equation, only current and previous steps exist)
    vector<double> u_curr(NX * NY, 0.0f);  // Current step
    vector<double> u_next(NX * NY, 0.0f);  // Next step

    // Initial condition: e.g., a Gaussian heat source at the center
    int cx = NX / 2;
    int cy = NY / 2;
    double sigma = 1.0f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            double x = i * dx;
            double y = j * dy;
            // Gaussian distribution
            u_curr[i * NY + j] = std::exp(-((x - Lx/2.0f)*(x - Lx/2.0f) + (y - Ly/2.0f)*(y - Ly/2.0f)) / (2.0f * sigma * sigma));
        }
    }

    dacpp::Matrix<double> matIn({NX, NY}, u_curr);
    dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);
    dacpp::Matrix<double> matOut = u_next_tensor[{1,NX-1}][{1,NY-1}];
    stencilShell_stencil(matIn, matOut);
    
    matIn[0].print();


    // Output some final result values as examples
    //cout << "Final temperature at center: " << vec2D[(NX/2)*NY + (NY/2)] << "\n";

    return 0;
}
