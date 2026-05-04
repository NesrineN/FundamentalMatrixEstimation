#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <cmath>
#include <vector>
#include "FNS.h"

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

Mat ComputeN(const Vec& u, const Mat& Eall, const std::vector<Mat>& Vall);
Mat Renorm(const Vec& u, const Mat& Eall, const std::vector<Mat>& Vall);
