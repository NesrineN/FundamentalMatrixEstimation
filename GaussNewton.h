#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <cmath>
#include "FNS.h"

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

Mat ComputeP(const Vec& u);

Mat computePseudoInverse(const Mat& A);

Mat GaussNewton(const Vec& u, const Mat& Eall);
