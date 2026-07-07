#include "libOrsa/libNumerics/matrix.h"
#include <vector>

#pragma once

struct Point2D
{
    double x;
    double y;
};


typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

int Triangulate(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime, const Mat& R, const Vec& t);
Mat Normaliza_Mat(const Mat& A);
double ReprojectionError(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime);
Mat EstimatePose(const Mat& K1, const Mat& K2, const Mat& F, const std::vector<Point2D>& img1Pts, const std::vector<Point2D>& img2Pts);
