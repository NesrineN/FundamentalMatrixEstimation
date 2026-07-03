#include "libOrsa/libNumerics/matrix.h"
#include <vector>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

struct Point2D
{
    double x;
    double y;
};

int Triangulate(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime);
Mat Normaliza_Mat(const Mat& A);
Mat EstimatePose(const Mat& K1, const Mat& K2, const Mat& F, const std::vector<Point2D>& img1Pts, const std::vector<Point2D>& img2Pts);
