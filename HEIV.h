#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

Mat ComputeV0Z(const Vec& E);

Mat ComputeMTilde(const Vec& v, const Mat& Zall, const Mat& Ztall);

Mat ComputeLTilde(const Vec& v, const Mat& Zall, const Mat& Ztall);

Vec solveGeneralizedEigen(const Mat& Mt, const Mat& Lt);

Mat HEIV(const Vec& v, const Mat& Zall, const Mat& Ztall, Vec Zbar, double f0);
