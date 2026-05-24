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
shell dacpp::list PDE(const dacpp::Vector<double>& u_kin,
                        dacpp::Vector<double>& u_kout,
                        const dacpp::Vector<double>& r) {
    dacpp::index i;
    dacpp::split s(3,1);
    binding(i, s);
    dacpp::list dataList{u_kin[s], u_kout[i], r[{}]};
    return dataList;
}

calc void pde(dacpp::Vector<double>& u_kin,
    double* u_kout,
    double* r) {
    u_kout[0] = r[0] * u_kin[0] + (1 - 2 * r[0]) * u_kin[1] + r[0] * u_kin[2];
}


int main() {
    int n = 5000; // Divide time domain into n steps
    int m = 30; // Divide space domain into m steps
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
        PDE(u_kin, u_kout, r) <-> pde;
        
        // After computation, replace points 1 to m-1
        for (int i = 1; i <= m-1; i++) {
            u_tensor[i][k+1] = u_kout[i-1];
        }

    }
    u_tensor[1].print();
    return 0;
}