#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"
using namespace std;

namespace dacpp {
    typedef std::vector<std::any> list;
}

// =====================
// Global physical parameters
// =====================
const int NX = 256;
const int NY = 256;
const int NZ = 256;
const double Lx = 10.0;
const double Ly = 10.0;
const double Lz = 10.0;
const double alpha = 0.01;
const int TIME_STEPS = 100;

const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);
const double dz = Lz / (NZ - 1);

// 3D stability condition
const double dt_stability = 1.0 / (2.0 * alpha * (1.0/(dx*dx) + 1.0/(dy*dy) + 1.0/(dz*dz)));
const double delta_t = 0.4 * dt_stability;

// =====================
// Shell: 3D window mapping
// =====================


// =====================
// Calc: 7-point operator computation
// =====================


#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void stencil3D_calc(double* blockIn,double* res,int blockIn_0,int blockIn_1,int blockIn_2,int res_0,int res_1,int res_2,int blockIn_0_shape,int blockIn_1_shape,int blockIn_2_shape,int res_0_shape,int res_1_shape,int res_2_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_blockIn_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_res_acc) {
    double center = blockIn[(1 + blockIn_0) * blockIn_1_shape * blockIn_2_shape + (1 + blockIn_1) * blockIn_2_shape + (1 + blockIn_2)];
    double d2u_dx2 = (blockIn[(2 + blockIn_0) * blockIn_1_shape * blockIn_2_shape + (1 + blockIn_1) * blockIn_2_shape + (1 + blockIn_2)]- 2. * center + blockIn[(0 + blockIn_0) * blockIn_1_shape * blockIn_2_shape + (1 + blockIn_1) * blockIn_2_shape + (1 + blockIn_2)]) / (dx * dx);
    double d2u_dy2 = (blockIn[(1 + blockIn_0) * blockIn_1_shape * blockIn_2_shape + (2 + blockIn_1) * blockIn_2_shape + (1 + blockIn_2)]- 2. * center + blockIn[(1 + blockIn_0) * blockIn_1_shape * blockIn_2_shape + (0 + blockIn_1) * blockIn_2_shape + (1 + blockIn_2)]) / (dy * dy);
    double d2u_dz2 = (blockIn[(1 + blockIn_0) * blockIn_1_shape * blockIn_2_shape + (1 + blockIn_1) * blockIn_2_shape + (2 + blockIn_2)]- 2. * center + blockIn[(1 + blockIn_0) * blockIn_1_shape * blockIn_2_shape + (1 + blockIn_1) * blockIn_2_shape + (0 + blockIn_2)]) / (dz * dz);
    res[(0 + res_0) * res_1_shape * res_2_shape + (0 + res_1) * res_2_shape + (0 + res_2)]= center + alpha * delta_t * (d2u_dx2 + d2u_dy2 + d2u_dz2);
}


void stencil3D_shell_stencil3D_calc(dacpp::Tensor<double, 3> & matIn, dacpp::Tensor<double, 3> & matOut) {
    auto selector = default_selector_v;
    queue q(selector);
    ParameterGeneration para_gene_tool;
    
    DataInfo info_matIn;
    info_matIn.dim = matIn.getDim();
    int info_matIn_Shape[3] = {0};
    for(int i = 0; i < info_matIn.dim; i++)
    {
        info_matIn.dimLength.push_back(matIn.getShape(i));
        info_matIn_Shape[i] = matIn.getShape(i);
    }
	
    DataInfo info_matOut;
    info_matOut.dim = matOut.getDim();
    int info_matOut_Shape[3] = {0};
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

    RegularSlice sp3 = RegularSlice("sp3", 3, 1);
    sp3.setDimId(2);
    sp3.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sp3,info_matIn));

    Index idx1 = Index("idx1");
    idx1.setDimId(0);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_matOut));

    Index idx2 = Index("idx2");
    idx2.setDimId(1);
    idx2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx2,info_matOut));

    Index idx3 = Index("idx3");
    idx3.setDimId(2);
    idx3.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx3,info_matOut));

	
	
    Dac_Ops matIn_Ops;
    
    sp1.setDimId(0);
    matIn_Ops.push_back(sp1);

    sp2.setDimId(1);
    matIn_Ops.push_back(sp2);

    sp3.setDimId(2);
    matIn_Ops.push_back(sp3);


    Dac_Ops matOut_Ops;
    
    idx1.setDimId(0);
    matOut_Ops.push_back(idx1);

    idx2.setDimId(1);
    matOut_Ops.push_back(idx2);

    idx3.setDimId(2);
    matOut_Ops.push_back(idx3);


    Dac_Ops In_Ops;
    
    sp1.setDimId(0);
    In_Ops.push_back(sp1);

    sp2.setDimId(1);
    In_Ops.push_back(sp2);

    sp3.setDimId(2);
    In_Ops.push_back(sp3);


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
    sp3.setDimId(2);
    matIn_ops.push_back(sp3);


    auto r_matIn = std::make_unique<sycl::buffer<double, 1>>(h_matIn,sycl::range<1>(matIn.getSize()));
    r_matIn->set_final_data(h_matIn);

	std::vector<int> info_partition_matIn=para_gene_tool.init_partition_data_shape(info_matIn,matIn_ops);
    sycl::buffer<int> info_partition_matIn_buffer(info_partition_matIn.data(), sycl::range<1>(info_partition_matIn.size()));

    
    Dac_Ops matOut_ops;
    
    idx1.setDimId(0);
    matOut_ops.push_back(idx1);
    idx2.setDimId(1);
    matOut_ops.push_back(idx2);
    idx3.setDimId(2);
    matOut_ops.push_back(idx3);


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

    for (int t = 0; t < TIME_STEPS; ++t) {
        // Execute parallel Stencil computation
        
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
			
            const auto sp1_=(item_id/sp2.split_size/sp3.split_size+(0))%sp1.split_size;
            const auto idx1_=(item_id/sp2.split_size/sp3.split_size+(0))%idx1.split_size;
            const auto sp2_=(item_id/sp3.split_size+(0))%sp2.split_size;
            const auto idx2_=(item_id/sp3.split_size+(0))%idx2.split_size;
            const auto sp3_=(item_id+(0))%sp3.split_size;
            const auto idx3_=(item_id+(0))%idx3.split_size;
			
			const auto matIn_0 = sp1_ * sp1.stride;
			const auto matIn_1 = sp2_ * sp2.stride;
			const auto matIn_2 = sp3_ * sp3.stride;
			const auto matOut_0 = idx1_ * idx1.stride;
			const auto matOut_1 = idx2_ * idx2.stride;
			const auto matOut_2 = idx3_ * idx3.stride;
            
            auto* d_matIn = acc_matIn.get_multi_ptr<access::decorated::no>().get();
            auto* d_matOut = acc_matOut.get_multi_ptr<access::decorated::no>().get();
			
            stencil3D_calc(d_matIn,d_matOut,matIn_0,matIn_1,matIn_2,matOut_0,matOut_1,matOut_2,info_matIn_Shape[0],info_matIn_Shape[1],info_matIn_Shape[2],info_matOut_Shape[0],info_matOut_Shape[1],info_matOut_Shape[2],info_partition_matIn_accessor,info_partition_matOut_accessor);
        });
    }).wait();

;

        // Handle boundary conditions and sync back to matIn (following 2D dacpp style)
        {
    int __iL = (0);
    int __iR = (NX);
    int __iN = __iR - __iL;
    int __jL = (0);
    int __jR = (NY);
    int __jN = __jR - __jL;
    q.submit([&](sycl::handler& h){
    auto acc_matIn = r_matIn->get_access<sycl::access::mode::read_write>(h);
    auto acc_matOut = r_matOut->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<2>(__iN, __jN), [=](sycl::id<2> idx){
      auto* d_matIn = acc_matIn.template get_multi_ptr<sycl::access::decorated::no>().get();
      auto* d_matOut = acc_matOut.template get_multi_ptr<sycl::access::decorated::no>().get();
      int i = __iL + idx[0];
      int j = __jL + idx[1];
      {
                for (int k = 0; k < NZ; k++) {
                    // Update interior points from *d_matOut
                    if (i > 0 && i < NX - 1 && j > 0 && j < NY - 1 && k > 0 && k < NZ - 1) {
                        d_matIn[(i) * info_matIn_Shape[1] * info_matIn_Shape[2] + (j) * info_matIn_Shape[2] + (k)]= d_matOut[(i - 1) * info_matOut_Shape[1] * info_matOut_Shape[2] + (j - 1) * info_matOut_Shape[2] + (k - 1)];
                    }
                }
            }
    });
  });
}

        
        // Boundary handling (6 faces)
        {
    int __iL = (0);
    int __iR = (NY);
    int __iN = __iR - __iL;
    int __jL = (0);
    int __jR = (NZ);
    int __jN = __jR - __jL;
    q.submit([&](sycl::handler& h){
    auto acc_matIn = r_matIn->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<2>(__iN, __jN), [=](sycl::id<2> idx){
      auto* d_matIn = acc_matIn.template get_multi_ptr<sycl::access::decorated::no>().get();
      int j = __iL + idx[0];
      int k = __jL + idx[1];
      {
                d_matIn[(0) * info_matIn_Shape[1] * info_matIn_Shape[2] + (j) * info_matIn_Shape[2] + (k)]= d_matIn[(1) * info_matIn_Shape[1] * info_matIn_Shape[2] + (j) * info_matIn_Shape[2] + (k)];
                d_matIn[(NX-1) * info_matIn_Shape[1] * info_matIn_Shape[2] + (j) * info_matIn_Shape[2] + (k)]= d_matIn[(NX-2) * info_matIn_Shape[1] * info_matIn_Shape[2] + (j) * info_matIn_Shape[2] + (k)];
            }
    });
  });
}

        {
    int __iL = (0);
    int __iR = (NX);
    int __iN = __iR - __iL;
    int __jL = (0);
    int __jR = (NZ);
    int __jN = __jR - __jL;
    q.submit([&](sycl::handler& h){
    auto acc_matIn = r_matIn->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<2>(__iN, __jN), [=](sycl::id<2> idx){
      auto* d_matIn = acc_matIn.template get_multi_ptr<sycl::access::decorated::no>().get();
      int i = __iL + idx[0];
      int k = __jL + idx[1];
      {
                d_matIn[(i) * info_matIn_Shape[1] * info_matIn_Shape[2] + (0) * info_matIn_Shape[2] + (k)]= d_matIn[(i) * info_matIn_Shape[1] * info_matIn_Shape[2] + (1) * info_matIn_Shape[2] + (k)];
                d_matIn[(i) * info_matIn_Shape[1] * info_matIn_Shape[2] + (NY-1) * info_matIn_Shape[2] + (k)]= d_matIn[(i) * info_matIn_Shape[1] * info_matIn_Shape[2] + (NY-2) * info_matIn_Shape[2] + (k)];
            }
    });
  });
}

        {
    int __iL = (0);
    int __iR = (NX);
    int __iN = __iR - __iL;
    int __jL = (0);
    int __jR = (NY);
    int __jN = __jR - __jL;
    q.submit([&](sycl::handler& h){
    auto acc_matIn = r_matIn->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<2>(__iN, __jN), [=](sycl::id<2> idx){
      auto* d_matIn = acc_matIn.template get_multi_ptr<sycl::access::decorated::no>().get();
      int i = __iL + idx[0];
      int j = __jL + idx[1];
      {
                d_matIn[(i) * info_matIn_Shape[1] * info_matIn_Shape[2] + (j) * info_matIn_Shape[2] + (0)]= d_matIn[(i) * info_matIn_Shape[1] * info_matIn_Shape[2] + (j) * info_matIn_Shape[2] + (1)];
                d_matIn[(i) * info_matIn_Shape[1] * info_matIn_Shape[2] + (j) * info_matIn_Shape[2] + (NZ-1)]= d_matIn[(i) * info_matIn_Shape[1] * info_matIn_Shape[2] + (j) * info_matIn_Shape[2] + (NZ-2)];
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
    // 1. Initialize data
    vector<double> u_curr(NX * NY * NZ, 0.0);
    vector<double> u_next_data(NX * NY * NZ, 0.0);

    double cx = Lx / 2.0, cy = Ly / 2.0, cz = Lz / 2.0;
    double sigma = 1.0;
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                double x = i * dx, y = j * dy, z = k * dz;
                double dist_sq = (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz);
                u_curr[i * NY * NZ + j * NZ + k] = std::exp(-dist_sq / (2.0 * sigma * sigma));
            }
        }
    }

    // 2. Construct dacpp Tensor objects
    dacpp::Tensor<double, 3> matIn({NX, NY, NZ}, u_curr);
    dacpp::Tensor<double, 3> u_next_tensor({NX, NY, NZ}, u_next_data);
    
    // Compute only for the interior region (Slice)
    dacpp::Tensor<double, 3> matOut = u_next_tensor[{1, NX-1}][{1, NY-1}][{1, NZ-1}];

    // 3. Time step iteration
    stencil3D_shell_stencil3D_calc(matIn, matOut);
    

    // 4. Output results
    // std::cout << "{";
    // for (int i = 0; i < NX; i++) {
    //     std::cout << "{";
    //     for (int j = 0; j < NY; j++) {
    //         std::cout << "{";
    //         for (int k = 0; k < NZ; k++) {
    //             std::cout << matIn[i][j][k];
    //             if (k < NZ - 1) std::cout << ", ";
    //         }
    //         std::cout << "}";
    //         if (j < NY - 1) std::cout << ", ";
    //     }
    //     std::cout << "}";
    //     if (i < NX - 1) std::cout << ", ";
    // }
    // std::cout << "}" << std::endl;

    matIn.print();
    return 0;
}