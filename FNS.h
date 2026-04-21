#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <cmath>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

Mat ComputeV0(const Vec& E);
Mat ComputeM(const Vec& u, const Mat& Eall);
Mat ComputeL(const Vec& u, const Mat& Eall);
Mat FNS(const Vec& u, const Mat& Eall);
