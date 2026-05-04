#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <cmath>
#include <vector>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

Mat ComputeM(const Vec& u, const Mat& Eall, const std::vector<Mat>& Vall);
Mat ComputeL(const Vec& u, const Mat& Eall, const std::vector<Mat>& Vall);
Vec SolveEigen(const Mat& A);
Mat FNS(const Vec& u, const Mat& Eall, const std::vector<Mat>& Vall);
