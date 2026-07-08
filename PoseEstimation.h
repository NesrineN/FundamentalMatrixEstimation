#include "libOrsa/libNumerics/matrix.h"
#include <vector>
#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>

using namespace Imagine;

#pragma once

struct Point2D
{
    double x;
    double y;
};


typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

int Triangulate(Image<Color,2> I1, Image<Color,2> I2, const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime, const Mat& R, const Vec& t);
double ReprojectionError(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime);
Mat Normaliza_Mat(const Mat& A);
void reorderForEssentialDecomposition(Mat& U, Vec& S, Mat& V);
double compareE(const Mat& E_est, const Mat& E_gt);
Mat EstimatePose(Image<Color,2> I1, Image<Color,2> I2, const Mat& K1, const Mat& K2, const Mat& F, const std::vector<Point2D>& img1Pts, const std::vector<Point2D>& img2Pts, const Mat& R_gt, const Vec& t_gt);
