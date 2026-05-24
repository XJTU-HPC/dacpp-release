#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"
using namespace std;

namespace dacpp {
    typedef std::vector<std::any> list;
}
// Grid parameters

const int NX = 8;    // Number of grid points in x direction
const int NY = 8;    // Number of grid points in y direction
const double Lx = 10.0f; // Length in x direction
const double Ly = 10.0f; // Length in y direction
const double c = 1.0f;   // Wave speed
const int TIME_STEPS = 10; // Number of time steps
// Grid spacing
const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);

// CFL condition
const double dt = 0.5f * std::fmin(dx, dy) / c; // Satisfy stability condition





#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void waveEq(double* cur,double* prev,double* next,int cur_0,int cur_1,int prev_0,int prev_1,int next_0,int next_1,int cur_0_shape,int cur_1_shape,int prev_0_shape,int prev_1_shape,int next_0_shape,int next_1_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_cur_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_prev_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_next_acc) {
    double dt = 0.5F * std::fmin(dx, dy) / c;
    double u_xx = (cur[(2 + cur_0) * cur_1_shape + (1 + cur_1)] - 2.F * cur[(1 + cur_0) * cur_1_shape + (1 + cur_1)] + cur[(0 + cur_0) * cur_1_shape + (1 + cur_1)]) / (dx * dx);
    double u_yy = (cur[(1 + cur_0) * cur_1_shape + (2 + cur_1)] - 2.F * cur[(1 + cur_0) * cur_1_shape + (1 + cur_1)] + cur[(1 + cur_0) * cur_1_shape + (0 + cur_1)]) / (dy * dy);
    next[(0 + next_0) * next_1_shape + (0 + next_1)] = 2.F * cur[(1 + cur_0) * cur_1_shape + (1 + cur_1)] - prev[(0 + prev_0) * prev_1_shape + (0 + prev_1)] + (c * c) * dt * dt * (u_xx + u_yy);
}


void waveEqShell_waveEq(dacpp::Matrix<double> & matCur, dacpp::Matrix<double> & matPrev, dacpp::Matrix<double> & matNext) {
    auto selector = default_selector_v;
    queue q(selector);
    ParameterGeneration para_gene_tool;
    
    DataInfo info_matCur;
    info_matCur.dim = matCur.getDim();
    int info_matCur_Shape[2] = {0};
    for(int i = 0; i < info_matCur.dim; i++)
    {
        info_matCur.dimLength.push_back(matCur.getShape(i));
        info_matCur_Shape[i] = matCur.getShape(i);
    }
	
    DataInfo info_matPrev;
    info_matPrev.dim = matPrev.getDim();
    int info_matPrev_Shape[2] = {0};
    for(int i = 0; i < info_matPrev.dim; i++)
    {
        info_matPrev.dimLength.push_back(matPrev.getShape(i));
        info_matPrev_Shape[i] = matPrev.getShape(i);
    }
	
    DataInfo info_matNext;
    info_matNext.dim = matNext.getDim();
    int info_matNext_Shape[2] = {0};
    for(int i = 0; i < info_matNext.dim; i++)
    {
        info_matNext.dimLength.push_back(matNext.getShape(i));
        info_matNext_Shape[i] = matNext.getShape(i);
    }
	
    RegularSlice sp1 = RegularSlice("sp1", 3, 1);
    sp1.setDimId(0);
    sp1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sp1,info_matCur));

    RegularSlice sp2 = RegularSlice("sp2", 3, 1);
    sp2.setDimId(1);
    sp2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sp2,info_matCur));

    Index idx1 = Index("idx1");
    idx1.setDimId(0);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_matPrev));

    Index idx2 = Index("idx2");
    idx2.setDimId(1);
    idx2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx2,info_matPrev));

	
	
    Dac_Ops matCur_Ops;
    
    sp1.setDimId(0);
    matCur_Ops.push_back(sp1);

    sp2.setDimId(1);
    matCur_Ops.push_back(sp2);


    Dac_Ops matPrev_Ops;
    
    idx1.setDimId(0);
    matPrev_Ops.push_back(idx1);

    idx2.setDimId(1);
    matPrev_Ops.push_back(idx2);


    Dac_Ops matNext_Ops;
    
    idx1.setDimId(0);
    matNext_Ops.push_back(idx1);

    idx2.setDimId(1);
    matNext_Ops.push_back(idx2);


    Dac_Ops In_Ops;
    
    sp1.setDimId(0);
    In_Ops.push_back(sp1);

    sp2.setDimId(1);
    In_Ops.push_back(sp2);


    Dac_Ops Out_Ops;
    
    idx1.setDimId(0);
    Out_Ops.push_back(idx1);

    idx2.setDimId(1);
    Out_Ops.push_back(idx2);


	
	
	
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    
    
	
    double* h_matCur = (double*)malloc(matCur.getSize()*sizeof(double));
    matCur.tensor2Array(h_matCur);

    double* h_matPrev = (double*)malloc(matPrev.getSize()*sizeof(double));
    matPrev.tensor2Array(h_matPrev);

    double* h_matNext = (double*)malloc(matNext.getSize()*sizeof(double));

    
    Dac_Ops matCur_ops;
    
    sp1.setDimId(0);
    matCur_ops.push_back(sp1);
    sp2.setDimId(1);
    matCur_ops.push_back(sp2);


    auto r_matCur = std::make_unique<sycl::buffer<double, 1>>(h_matCur,sycl::range<1>(matCur.getSize()));
    r_matCur->set_final_data(h_matCur);

	std::vector<int> info_partition_matCur=para_gene_tool.init_partition_data_shape(info_matCur,matCur_ops);
    sycl::buffer<int> info_partition_matCur_buffer(info_partition_matCur.data(), sycl::range<1>(info_partition_matCur.size()));

    
    Dac_Ops matPrev_ops;
    
    idx1.setDimId(0);
    matPrev_ops.push_back(idx1);
    idx2.setDimId(1);
    matPrev_ops.push_back(idx2);


    auto r_matPrev = std::make_unique<sycl::buffer<double, 1>>(h_matPrev,sycl::range<1>(matPrev.getSize()));
    r_matPrev->set_final_data(h_matPrev);

	std::vector<int> info_partition_matPrev=para_gene_tool.init_partition_data_shape(info_matPrev,matPrev_ops);
    sycl::buffer<int> info_partition_matPrev_buffer(info_partition_matPrev.data(), sycl::range<1>(info_partition_matPrev.size()));

    
    Dac_Ops matNext_ops;
    
    idx1.setDimId(0);
    matNext_ops.push_back(idx1);
    idx2.setDimId(1);
    matNext_ops.push_back(idx2);


    auto r_matNext = std::make_unique<sycl::buffer<double, 1>>(h_matNext,sycl::range<1>(matNext.getSize()));
    r_matNext->set_final_data(h_matNext);

	std::vector<int> info_partition_matNext=para_gene_tool.init_partition_data_shape(info_matNext,matNext_ops);
    sycl::buffer<int> info_partition_matNext_buffer(info_partition_matNext.data(), sycl::range<1>(info_partition_matNext.size()));

	
	
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

    for(int i = 0;i < TIME_STEPS; i++) {
        
    q.submit([&](handler &h) {
    
        accessor<double, 1, sycl::access::mode::read_write> acc_matCur(*r_matCur, h);
        accessor<double, 1, sycl::access::mode::read_write> acc_matPrev(*r_matPrev, h);
        accessor<double, 1, sycl::access::mode::discard_write> acc_matNext(*r_matNext, h);
    
        auto info_partition_matCur_accessor = info_partition_matCur_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_matPrev_accessor = info_partition_matPrev_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_matNext_accessor = info_partition_matNext_buffer.get_access<sycl::access::mode::read>(h);
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
			
			const auto matCur_0 = sp1_ * sp1.stride;
			const auto matCur_1 = sp2_ * sp2.stride;
			const auto matPrev_0 = idx1_ * idx1.stride;
			const auto matPrev_1 = idx2_ * idx2.stride;
			const auto matNext_0 = idx1_ * idx1.stride;
			const auto matNext_1 = idx2_ * idx2.stride;
            
            auto* d_matCur = acc_matCur.get_multi_ptr<access::decorated::no>().get();
            auto* d_matPrev = acc_matPrev.get_multi_ptr<access::decorated::no>().get();
            auto* d_matNext = acc_matNext.get_multi_ptr<access::decorated::no>().get();
			
            waveEq(d_matCur,d_matPrev,d_matNext,matCur_0,matCur_1,matPrev_0,matPrev_1,matNext_0,matNext_1,info_matCur_Shape[0],info_matCur_Shape[1],info_matPrev_Shape[0],info_matPrev_Shape[1],info_matNext_Shape[0],info_matNext_Shape[1],info_partition_matCur_accessor,info_partition_matPrev_accessor,info_partition_matNext_accessor);
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
    auto acc_matCur = r_matCur->get_access<sycl::access::mode::read_write>(h);
    auto acc_matPrev = r_matPrev->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<2>(__iN, __jN), [=](sycl::id<2> idx){
      auto* d_matCur = acc_matCur.template get_multi_ptr<sycl::access::decorated::no>().get();
      auto* d_matPrev = acc_matPrev.template get_multi_ptr<sycl::access::decorated::no>().get();
      int i = __iL + idx[0];
      int j = __jL + idx[1];
      {
                d_matPrev[(i-1) * info_matPrev_Shape[1] + (j-1)]=d_matCur[(i) * info_matCur_Shape[1] + (j)];
            }
    });
  });
}


        {
    int __iL = (1);
    int __iR = (NX-2);
    int __iN = __iR - __iL + 1;
    int __jL = (1);
    int __jR = (NY-2);
    int __jN = __jR - __jL + 1;
    q.submit([&](sycl::handler& h){
    auto acc_matCur = r_matCur->get_access<sycl::access::mode::read_write>(h);
    auto acc_matNext = r_matNext->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<2>(__iN, __jN), [=](sycl::id<2> idx){
      auto* d_matCur = acc_matCur.template get_multi_ptr<sycl::access::decorated::no>().get();
      auto* d_matNext = acc_matNext.template get_multi_ptr<sycl::access::decorated::no>().get();
      int i = __iL + idx[0];
      int j = __jL + idx[1];
      {
                d_matCur[(i) * info_matCur_Shape[1] + (j)]=d_matNext[(i-1) * info_matNext_Shape[1] + (j-1)];
            }
    });
  });
}

        // Handle boundary conditions (adiabatic boundary: zero derivative)
        {
  int __L = (0);
  int __R = (NX-1);
  int __N = __R - __L + 1;
  q.submit([&](sycl::handler& h){
    auto acc_matCur = r_matCur->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>(__N), [=](sycl::id<1> idx){
      auto* d_matCur = acc_matCur.template get_multi_ptr<sycl::access::decorated::no>().get();
      int i = __L + idx[0];
      {       
            d_matCur[(i) * info_matCur_Shape[1] + (NY-1)]=0;
            d_matCur[(i) * info_matCur_Shape[1] + (0)]=0;
        }
    });
  });
}

        {
  int __L = (0);
  int __R = (NY-1);
  int __N = __R - __L + 1;
  q.submit([&](sycl::handler& h){
    auto acc_matCur = r_matCur->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>(__N), [=](sycl::id<1> idx){
      auto* d_matCur = acc_matCur.template get_multi_ptr<sycl::access::decorated::no>().get();
      int j = __L + idx[0];
      {
            d_matCur[(NX - 1) * info_matCur_Shape[1] + (j)]=0;
            d_matCur[(0) * info_matCur_Shape[1] + (j)]=0;
             // Bottom boundary
        }
    });
  });
}

    }


	
    r_matCur.reset();
    matCur.array2Tensor(h_matCur);

    r_matPrev.reset();
    matPrev.array2Tensor(h_matPrev);

    r_matNext.reset();
    matNext.array2Tensor(h_matNext);

	

}

int main() {
    // Initialize wave field
    vector<double> u_prev(NX * NY, 0.0f); // Previous step
    vector<double> u_curr(NX * NY, 0.0f);  // Current step
    vector<double> u_next(NX * NY, 0.0f);  // Current step

    // Initial condition: e.g., a Gaussian pulse
    int cx = NX / 2;
    int cy = NY / 2;
    double sigma = 0.5f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            double x = i * dx;
            double y = j * dy;
            u_prev[i*NX+j] = std::exp(-((x - Lx/2)*(x - Lx/2) + (y - Ly/2)*(y - Ly/2)) / (2 * sigma * sigma));
        }
    }

    dacpp::Matrix<double> matCur({NX, NY}, u_curr);
    dacpp::Matrix<double> u_prev_tensor({NX, NY}, u_prev);
    dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);
    dacpp::Matrix<double> matPrev = u_prev_tensor[{1,NX-1}][{1,NY-1}];
    dacpp::Matrix<double> matNext = u_next_tensor[{1,NX-1}][{1,NY-1}];
    
    waveEqShell_waveEq(matCur, matPrev, matNext);
    
    //
    matCur.print(); 
    return 0;
}