#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <cmath>
#include "FNS.h"

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

Mat ComputeN(const Vec& u, const Mat& Eall);
Mat Renorm(const Vec& u, const Mat& Eall);
