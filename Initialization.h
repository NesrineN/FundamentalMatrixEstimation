#include "libOrsa/libNumerics/matrix.h"
#include "HEIV.h"
#include <iostream> 
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

Vec RandomInit();
Mat ComputeMLS(const Mat& Eall);
Mat ComputeMLSt(const Mat& Eall, const Vec& Zbar);
Mat ComputeNTBt(const Mat& Eall);
Vec LeastSquares(const Mat& Eall);
Vec Taubin(const Mat& Eall);