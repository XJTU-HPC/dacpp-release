#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <any>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <queue>
#include "ReconTensor.h"
#define DACPP_TRANSLATE_MODE 1

// Vector-related operations need to be implemented
namespace dacpp {
    typedef std::vector<std::any> list;
}

double phi(double x) { return x*x*x+x; }

double alpha(double t) { return 0.0; }

double beta(double t) { return 1.0+exp(t); }

double f(double x, double t) { return x*exp(t)-6*x; }

double exact(double x, double t) { return x*(x*x+exp(t)); }

// Same issue: during partitioning, one data to compute and three computation data, four data total to partition together




#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void pde(const double* u_kin,double* u_kout,const double* r,int u_kin_0,int u_kout_0,int r_0,int u_kin_0_shape,int u_kout_0_shape,int r_0_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_u_kin_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_u_kout_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_r_acc) {
    u_kout[0+u_kout_0] = r[0+r_0] * u_kin[0+u_kin_0] + (1 - 2 * r[0+r_0]) * u_kin[1+u_kin_0] + r[0+r_0] * u_kin[2+u_kin_0];
}


void PDE_pde(const dacpp::Vector<double> & u_kin, dacpp::Vector<double> & u_kout, const dacpp::Vector<double> & r) {
    auto selector = default_selector_v;
    queue q(selector);
    ParameterGeneration para_gene_tool;
    
    DataInfo info_u_kin;
    info_u_kin.dim = u_kin.getDim();
    int info_u_kin_Shape[1] = {0};
    for(int i = 0; i < info_u_kin.dim; i++)
    {
        info_u_kin.dimLength.push_back(u_kin.getShape(i));
        info_u_kin_Shape[i] = u_kin.getShape(i);
    }
	
    DataInfo info_u_kout;
    info_u_kout.dim = u_kout.getDim();
    int info_u_kout_Shape[1] = {0};
    for(int i = 0; i < info_u_kout.dim; i++)
    {
        info_u_kout.dimLength.push_back(u_kout.getShape(i));
        info_u_kout_Shape[i] = u_kout.getShape(i);
    }
	
    DataInfo info_r;
    info_r.dim = r.getDim();
    int info_r_Shape[1] = {0};
    for(int i = 0; i < info_r.dim; i++)
    {
        info_r.dimLength.push_back(r.getShape(i));
        info_r_Shape[i] = r.getShape(i);
    }
	
    RegularSlice s = RegularSlice("s", 3, 1);
    s.setDimId(0);
    s.SetSplitSize(para_gene_tool.init_operetor_splitnumber(s,info_u_kin));

    Index i = Index("i");
    i.setDimId(0);
    i.SetSplitSize(para_gene_tool.init_operetor_splitnumber(i,info_u_kout));

	
	
    Dac_Ops u_kin_Ops;
    
    s.setDimId(0);
    u_kin_Ops.push_back(s);


    Dac_Ops u_kout_Ops;
    
    i.setDimId(0);
    u_kout_Ops.push_back(i);


    Dac_Ops r_Ops;
    

    Dac_Ops In_Ops;
    
    s.setDimId(0);
    In_Ops.push_back(s);


    Dac_Ops Out_Ops;
    
    i.setDimId(0);
    Out_Ops.push_back(i);


	
	
	
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    
    
	
    double* h_u_kin = (double*)malloc(u_kin.getSize()*sizeof(double));
    u_kin.tensor2Array(h_u_kin);
	buffer<double, 1> r_u_kin(h_u_kin, range<1>(u_kin.getSize()));

    double* h_u_kout = (double*)malloc(u_kout.getSize()*sizeof(double));

    double* h_r = (double*)malloc(r.getSize()*sizeof(double));
    r.tensor2Array(h_r);
	buffer<double, 1> r_r(h_r, range<1>(r.getSize()));


    
    Dac_Ops u_kin_ops;
    
    s.setDimId(0);
    u_kin_ops.push_back(s);


	std::vector<int> info_partition_u_kin=para_gene_tool.init_partition_data_shape(info_u_kin,u_kin_ops);
    sycl::buffer<int> info_partition_u_kin_buffer(info_partition_u_kin.data(), sycl::range<1>(info_partition_u_kin.size()));

    
    Dac_Ops u_kout_ops;
    
    i.setDimId(0);
    u_kout_ops.push_back(i);


    auto r_u_kout = std::make_unique<sycl::buffer<double, 1>>(h_u_kout,sycl::range<1>(u_kout.getSize()));
    r_u_kout->set_final_data(h_u_kout);

	std::vector<int> info_partition_u_kout=para_gene_tool.init_partition_data_shape(info_u_kout,u_kout_ops);
    sycl::buffer<int> info_partition_u_kout_buffer(info_partition_u_kout.data(), sycl::range<1>(info_partition_u_kout.size()));


    
    Dac_Ops r_ops;
    


	std::vector<int> info_partition_r=para_gene_tool.init_partition_data_shape(info_r,r_ops);
    sycl::buffer<int> info_partition_r_buffer(info_partition_r.data(), sycl::range<1>(info_partition_r.size()));

	
	
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
    
        accessor<double, 1, access::mode::read> acc_u_kin(r_u_kin, h);
        r_u_kin.set_final_data(nullptr);
        
        accessor<double, 1, sycl::access::mode::discard_write> acc_u_kout(*r_u_kout, h);
        accessor<double, 1, access::mode::read> acc_r(r_r, h);
        r_r.set_final_data(nullptr);
        
    
        auto info_partition_u_kin_accessor = info_partition_u_kin_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_u_kout_accessor = info_partition_u_kout_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_r_accessor = info_partition_r_buffer.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
			
            const auto i_=(item_id+(0))%i.split_size;
            const auto s_=(item_id+(0))%s.split_size;
			
			const auto u_kin_0 = s_ * s.stride;
			const auto u_kout_0 = i_ * i.stride;
			const auto r_0 = 0;
            
            auto* d_u_kin = acc_u_kin.get_multi_ptr<access::decorated::no>().get();
            auto* d_u_kout = acc_u_kout.get_multi_ptr<access::decorated::no>().get();
            auto* d_r = acc_r.get_multi_ptr<access::decorated::no>().get();
			
            pde(d_u_kin,d_u_kout,d_r,u_kin_0,u_kout_0,r_0,info_u_kin_Shape[0],info_u_kout_Shape[0],info_r_Shape[0],info_partition_u_kin_accessor,info_partition_u_kout_accessor,info_partition_r_accessor);
        });
    }).wait();


	
    r_u_kout.reset();
    u_kout.array2Tensor(h_u_kout);

	

}

int main() {
    int n = 100; // Divide time domain into n steps
    int m = 5; // Divide space domain into m steps
    double r;
    double a = 1.0;
    double h = 1.0 / m; // Spatial step size
    double tau = 1.0 / n; // Time step size
    double *x,*t,**u;
    
    r=a*tau/(h*h);  // Mesh ratio
    //printf("r=%.4f.\n",r);
    
    x = (double*)malloc(sizeof(double)*(m+1));
    for (int i=0;i<=m;i++) {
        x[i]=i*h;
    }
    t = (double*)malloc(sizeof(double)*(n+1));
    for (int i = 0; i <= n; i++) {
        t[i]=i*tau;
    }
    u = (double**)malloc(sizeof(double*)*(m+1));
    for (int i=0;i<=m;i++) {
        u[i]=(double*)malloc(sizeof(double)*(n+1));
    }
    for (int i = 0; i <= m; i++)
        u[i][0]=phi(x[i]);
    for (int i = 1; i <= n; i++) {
        u[0][i]=alpha(t[i]);
        u[m][i]=beta(t[i]);
    }
    
    // Flatten the 2D u array into a 1D vector for Tensor creation
    std::vector<double> u_flat;
    for (int i = 0; i <= m; ++i) {
        for (int j = 0; j <= n; ++j) {
            u_flat.push_back(static_cast<double>(u[i][j]));  // Cast if needed
        }
    }

    dacpp::Matrix<double> u_tensor({m+1, n+1}, u_flat);
    for (int k = 0; k <= n-1; k++) {
        dacpp::Vector<double> u_kout = u_tensor[{1,m}][k+1];
        std::vector<double> r_data;
        r_data.push_back(r);
        dacpp::Vector<double> r(r_data);
        dacpp::Vector<double> u_kin = u_tensor[{}][k];
        PDE_pde(u_kin, u_kout, r);
        
        // After computation, replace points 1 to m-1
        for (int i = 1; i <= m-1; i++) {
            u_tensor[i][k+1] = u_kout[i-1];
        }

    }
    u_tensor[1].print();
    return 0;
}