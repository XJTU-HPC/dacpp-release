#include <iostream>
#include <vector>
#include <complex>
#include "ReconTensor.h"
#include <cmath>

namespace dacpp {
    typedef std::vector<std::any> list;
}
using namespace std;


// Global variable definitions
const int row_count = 8, col_count = 8, max_iterations = 1000;
vector<complex<float>> complex_points;  // One-dimensional vector representing complex points
vector<int> mandelbrot_flags;           // One-dimensional array indicating whether each point belongs to the Mandelbrot set
int total_points = 0;                   // Total number of points
int mandelbrot_count = 0;               // Number of points belonging to the Mandelbrot set

// Initialize complex points vector
void InitializeComplexPoints() {
    total_points = row_count * col_count;  // Total number of points
    complex_points.resize(total_points);

    for (int i = 0; i < row_count; ++i) {
        for (int j = 0; j < col_count; ++j) {
            int index = i * col_count + j;  // One-dimensional vector index
            float real = -1.5f + (i * (2.0f / row_count));  // Map row index to real part
            float imag = -1.0f + (j * (2.0f / col_count));  // Map column index to imaginary part
            complex_points[index] = complex<float>(real, imag);
        }
    }
}







// Print statistics
void PrintStats() {
    cout << "Mandelbrot Set Statistics:\n";
    cout << "Total points: " << total_points << "\n";
    cout << "Points in the Mandelbrot set: " << mandelbrot_count << "\n";
}

#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void mandel(const complex<float>* complex_points,int* mandelbrot_flags,int complex_points_0,int mandelbrot_flags_0,int complex_points_0_shape,int mandelbrot_flags_0_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_complex_points_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_mandelbrot_flags_acc) {
    const complex<float> &c = complex_points[0+complex_points_0];
    complex<float> z = 0;
    int iterations = 0;
    for (int i = 0; i < max_iterations; ++i) {
        if (std::sqrt(z.real() * z.real() + z.imag() * z.imag()) > 2.F) {
            iterations = i;
            break;
        }
        z = z * z + c;
        iterations = max_iterations;
    }
    if (iterations == max_iterations) {
        mandelbrot_flags[0+mandelbrot_flags_0] = 1;
    }
}


void MANDEL_mandel(const dacpp::Vector<complex<float> > & complex_points, dacpp::Vector<int> & mandelbrot_flags) {
    auto selector = default_selector_v;
    queue q(selector);
    ParameterGeneration para_gene_tool;
    
    DataInfo info_complex_points;
    info_complex_points.dim = complex_points.getDim();
    int info_complex_points_Shape[1] = {0};
    for(int i = 0; i < info_complex_points.dim; i++)
    {
        info_complex_points.dimLength.push_back(complex_points.getShape(i));
        info_complex_points_Shape[i] = complex_points.getShape(i);
    }
	
    DataInfo info_mandelbrot_flags;
    info_mandelbrot_flags.dim = mandelbrot_flags.getDim();
    int info_mandelbrot_flags_Shape[1] = {0};
    for(int i = 0; i < info_mandelbrot_flags.dim; i++)
    {
        info_mandelbrot_flags.dimLength.push_back(mandelbrot_flags.getShape(i));
        info_mandelbrot_flags_Shape[i] = mandelbrot_flags.getShape(i);
    }
	
    Index i = Index("i");
    i.setDimId(0);
    i.SetSplitSize(para_gene_tool.init_operetor_splitnumber(i,info_complex_points));

	
	
    Dac_Ops complex_points_Ops;
    
    i.setDimId(0);
    complex_points_Ops.push_back(i);


    Dac_Ops mandelbrot_flags_Ops;
    
    i.setDimId(0);
    mandelbrot_flags_Ops.push_back(i);


    Dac_Ops In_Ops;
    
    i.setDimId(0);
    In_Ops.push_back(i);


    Dac_Ops Out_Ops;
    
    i.setDimId(0);
    Out_Ops.push_back(i);


	
	
	
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    
    
	
    complex<float>* h_complex_points = (complex<float>*)malloc(complex_points.getSize()*sizeof(complex<float>));
    complex_points.tensor2Array(h_complex_points);
	buffer<complex<float>, 1> r_complex_points(h_complex_points, range<1>(complex_points.getSize()));

    int* h_mandelbrot_flags = (int*)malloc(mandelbrot_flags.getSize()*sizeof(int));


    
    Dac_Ops complex_points_ops;
    
    i.setDimId(0);
    complex_points_ops.push_back(i);


	std::vector<int> info_partition_complex_points=para_gene_tool.init_partition_data_shape(info_complex_points,complex_points_ops);
    sycl::buffer<int> info_partition_complex_points_buffer(info_partition_complex_points.data(), sycl::range<1>(info_partition_complex_points.size()));

    
    Dac_Ops mandelbrot_flags_ops;
    
    i.setDimId(0);
    mandelbrot_flags_ops.push_back(i);


    auto r_mandelbrot_flags = std::make_unique<sycl::buffer<int, 1>>(h_mandelbrot_flags,sycl::range<1>(mandelbrot_flags.getSize()));
    r_mandelbrot_flags->set_final_data(h_mandelbrot_flags);

	std::vector<int> info_partition_mandelbrot_flags=para_gene_tool.init_partition_data_shape(info_mandelbrot_flags,mandelbrot_flags_ops);
    sycl::buffer<int> info_partition_mandelbrot_flags_buffer(info_partition_mandelbrot_flags.data(), sycl::range<1>(info_partition_mandelbrot_flags.size()));

	
	
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
    
        accessor<complex<float>, 1, access::mode::read> acc_complex_points(r_complex_points, h);
        r_complex_points.set_final_data(nullptr);
        
        accessor<int, 1, sycl::access::mode::discard_write> acc_mandelbrot_flags(*r_mandelbrot_flags, h);
    
        auto info_partition_complex_points_accessor = info_partition_complex_points_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_mandelbrot_flags_accessor = info_partition_mandelbrot_flags_buffer.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
			
            const auto i_=(item_id+(0))%i.split_size;
			
			const auto complex_points_0 = i_ * i.stride;
			const auto mandelbrot_flags_0 = i_ * i.stride;
            
            auto* d_complex_points = acc_complex_points.get_multi_ptr<access::decorated::no>().get();
            auto* d_mandelbrot_flags = acc_mandelbrot_flags.get_multi_ptr<access::decorated::no>().get();
			
            mandel(d_complex_points,d_mandelbrot_flags,complex_points_0,mandelbrot_flags_0,info_complex_points_Shape[0],info_mandelbrot_flags_Shape[0],info_partition_complex_points_accessor,info_partition_mandelbrot_flags_accessor);
        });
    }).wait();


	
    r_mandelbrot_flags.reset();
    mandelbrot_flags.array2Tensor(h_mandelbrot_flags);

	

}

int main() {
    // Initialize complex points vector
    InitializeComplexPoints();

    // Compute the Mandelbrot set
    mandelbrot_flags.resize(total_points, 0);  // Initialize one-dimensional array to 0

    dacpp::Vector<complex<float>> complex_points_tensor(complex_points);
    dacpp::Vector<int> mandelbrot_flags_tensor(mandelbrot_flags);


    MANDEL_mandel(complex_points_tensor, mandelbrot_flags_tensor);

    // Count the number of 1s in the array
    mandelbrot_count = 0;
    for (int i = 0; i < total_points; i++){
        if (mandelbrot_flags_tensor[i] == 1) mandelbrot_count++;
    }

    // Print statistics
    PrintStats();

    return 0;
}
