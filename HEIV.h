#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

Mat ComputeV0Z(const Vec& E);

Vec ComputeZbar(const Vec& v, const Mat& Eall);

Mat ComputeMTilde(const Vec& v, const Mat& Eall, const Vec& Zbar);


Mat ComputeLTilde(const Vec& v, const Mat& Eall, const Vec& Zbar);

Vec SolveGeneralizedEigen(const Mat& Mt, const Mat& Lt);


Mat HEIV(const Vec& v, const Mat& Eall);