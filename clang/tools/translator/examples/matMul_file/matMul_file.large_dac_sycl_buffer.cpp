#include <iostream>
#include <vector>
#include "ReconTensor.h"
#include <string>
#include <filesystem>

namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;





#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void matrixMultiply_calc(const int* vecA,const int* vecB,int* dotProduct,int vecA_0,int vecA_1,int vecB_0,int vecB_1,int dotProduct_0,int dotProduct_1,int vecA_0_shape,int vecA_1_shape,int vecB_0_shape,int vecB_1_shape,int dotProduct_0_shape,int dotProduct_1_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_vecA_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_vecB_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_dotProduct_acc) {
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        sum += vecA[(0 + vecA_0) * vecA_1_shape + (i + vecA_1)] * vecB[(i + vecB_0) * vecB_1_shape + (0 + vecB_1)];
    }
    dotProduct[(0 + dotProduct_0) * dotProduct_1_shape + (0 + dotProduct_1)] = sum;
}


// Generate function call
void matrixMultiply_shell_matrixMultiply_calc(const dacpp::Matrix<int> & matA, const dacpp::Matrix<int> & matB, dacpp::Matrix<int> & matC) {
    auto selector = default_selector_v;
    queue q(selector);
    ParameterGeneration para_gene_tool;

    DataInfo info_matA;
    info_matA.dim = matA.getDim();
    int info_matA_Shape[2] = {0};
    for(int i = 0; i < info_matA.dim; i++)
    {
        info_matA.dimLength.push_back(matA.getShape(i));
        info_matA_Shape[i] = matA.getShape(i);
    }

    DataInfo info_matB;
    info_matB.dim = matB.getDim();
    int info_matB_Shape[2] = {0};
    for(int i = 0; i < info_matB.dim; i++)
    {
        info_matB.dimLength.push_back(matB.getShape(i));
        info_matB_Shape[i] = matB.getShape(i);
    }

    DataInfo info_matC;
    info_matC.dim = matC.getDim();
    int info_matC_Shape[2] = {0};
    for(int i = 0; i < info_matC.dim; i++)
    {
        info_matC.dimLength.push_back(matC.getShape(i));
        info_matC_Shape[i] = matC.getShape(i);
    }

    Index idx1 = Index("idx1");
    idx1.setDimId(0);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_matA));

    Index idx2 = Index("idx2");
    idx2.setDimId(1);
    idx2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx2,info_matB));

    Dac_Ops matA_Ops;

    idx1.setDimId(0);
    matA_Ops.push_back(idx1);


    Dac_Ops matB_Ops;

    idx2.setDimId(1);
    matB_Ops.push_back(idx2);


    Dac_Ops matC_Ops;

    idx1.setDimId(0);
    matC_Ops.push_back(idx1);

    idx2.setDimId(1);
    matC_Ops.push_back(idx2);


    Dac_Ops In_Ops;

    idx1.setDimId(0);
    In_Ops.push_back(idx1);

    idx2.setDimId(1);
    In_Ops.push_back(idx2);


    Dac_Ops Out_Ops;

    idx1.setDimId(0);
    Out_Ops.push_back(idx1);

    idx2.setDimId(1);
    Out_Ops.push_back(idx2);


    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    int* h_matA = (int*)malloc(matA.getSize()*sizeof(int));
    matA.tensor2Array(h_matA);
	buffer<int, 1> r_matA(h_matA, range<1>(matA.getSize()));

    int* h_matB = (int*)malloc(matB.getSize()*sizeof(int));
    matB.tensor2Array(h_matB);
	buffer<int, 1> r_matB(h_matB, range<1>(matB.getSize()));

    int* h_matC = (int*)malloc(matC.getSize()*sizeof(int));

    Dac_Ops matA_ops;

    idx1.setDimId(0);
    matA_ops.push_back(idx1);


	std::vector<int> info_partition_matA=para_gene_tool.init_partition_data_shape(info_matA,matA_ops);
    sycl::buffer<int> info_partition_matA_buffer(info_partition_matA.data(), sycl::range<1>(info_partition_matA.size()));

    Dac_Ops matB_ops;

    idx2.setDimId(1);
    matB_ops.push_back(idx2);


	std::vector<int> info_partition_matB=para_gene_tool.init_partition_data_shape(info_matB,matB_ops);
    sycl::buffer<int> info_partition_matB_buffer(info_partition_matB.data(), sycl::range<1>(info_partition_matB.size()));

    Dac_Ops matC_ops;

    idx1.setDimId(0);
    matC_ops.push_back(idx1);
    idx2.setDimId(1);
    matC_ops.push_back(idx2);


    auto r_matC = std::make_unique<sycl::buffer<int, 1>>(h_matC,sycl::range<1>(matC.getSize()));
    r_matC->set_final_data(h_matC);

	std::vector<int> info_partition_matC=para_gene_tool.init_partition_data_shape(info_matC,matC_ops);
    sycl::buffer<int> info_partition_matC_buffer(info_partition_matC.data(), sycl::range<1>(info_partition_matC.size()));

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

        accessor<int, 1, access::mode::read> acc_matA(r_matA, h);
        r_matA.set_final_data(nullptr);

        accessor<int, 1, access::mode::read> acc_matB(r_matB, h);
        r_matB.set_final_data(nullptr);

        accessor<int, 1, sycl::access::mode::discard_write> acc_matC(*r_matC, h);

        auto info_partition_matA_accessor = info_partition_matA_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_matB_accessor = info_partition_matB_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_matC_accessor = info_partition_matC_buffer.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;

            const auto idx1_=(item_id/idx2.split_size+(0))%idx1.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;

			const auto matA_0 = idx1_ * idx1.stride;
			const auto matA_1 = 0;
			const auto matB_0 = 0;
			const auto matB_1 = idx2_ * idx2.stride;
			const auto matC_0 = idx1_ * idx1.stride;
			const auto matC_1 = idx2_ * idx2.stride;

            auto* d_matA = acc_matA.get_multi_ptr<access::decorated::no>().get();
            auto* d_matB = acc_matB.get_multi_ptr<access::decorated::no>().get();
            auto* d_matC = acc_matC.get_multi_ptr<access::decorated::no>().get();

            matrixMultiply_calc(d_matA,d_matB,d_matC,matA_0,matA_1,matB_0,matB_1,matC_0,matC_1,info_matA_Shape[0],info_matA_Shape[1],info_matB_Shape[0],info_matB_Shape[1],info_matC_Shape[0],info_matC_Shape[1],info_partition_matA_accessor,info_partition_matB_accessor,info_partition_matC_accessor);
        });
    }).wait();

    r_matC.reset();
    matC.array2Tensor(h_matC);

}

int main() {
    std::string dir = std::filesystem::path(__FILE__).parent_path().string();

    auto objA = dacpp::read_matrix_mtx_col<int>(dir + "/dataA_large.mtx");
    auto objB = dacpp::read_matrix_mtx_col<int>(dir + "/dataB_large.mtx");

    dacpp::Matrix<int> matA(objA.shape, objA.data);
    dacpp::Matrix<int> matB(objB.shape, objB.data);

    // Even the size of matC can be dynamically derived
    std::vector<int> dataC(objA.shape[0] * objB.shape[1], 0);
    dacpp::Matrix<int> matC({objA.shape[0], objB.shape[1]}, dataC);

    matrixMultiply_shell_matrixMultiply_calc(matA, matB, matC);

    matC.print();

    return 0;
}
