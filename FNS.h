#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <cmath>
#include <vector>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

Mat ComputeM(const Vec& u, const Mat& Eall, const std::vector<Mat>& Vall);
Mat ComputeL(const Vec& u, const Mat& Eall, const std::vector<Mat>& Vall);
void EigenSVD(const Mat& A, Mat& U, Vec& S, Mat& V);
void EigenSVD3x3(const Mat& A, Mat& U, Vec& S, Mat& V);
Vec SVD_U(const Mat& A);
Mat FNS(const Vec& u, const Mat& Eall, const std::vector<Mat>& Vall);
Vec SolveEigen(const Mat& A);
