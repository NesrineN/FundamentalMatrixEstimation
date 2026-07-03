#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <cmath>
#include "FNS.h"
#include "HEIV.h"
#include "RANSAC.h"
#include "Renorm.h"
#include "GaussNewton.h"
#include "Initialization.h"
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include "PoseEstimation.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

const double PI = 3.14159265358979323846;

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

using namespace Imagine;
using namespace std;

struct Point2D
{
    double x;
    double y;
};

// function that creates the vector E from two point-correspondences:
Vec fillE(const Point2D& p1, const Point2D& p2, double f0);

Mat computeV0(double x, double y, double xp, double yp, double f0);

Mat GetF(const std::vector<Point2D>& img1Pts, const std::vector<Point2D>& img2Pts, const double& f0, const int& method);

void printM(const Mat& A);

void printV(const Vec& V);

double trace(const Mat& R);

inline double clamp(double x, double a, double b);

double norm(const Vec& v);

double rotation_error(const Mat& R_gt, const Mat& R_pred);

double translation_error(const Vec& t_gt, const Vec& t_pred);

Vec RunPipelineNoiseless(const std::string& I1_path, const std::string& I2_path, const Mat& K1, const Mat& K2, const double& f0, const int& method=1, const Mat& R_gt, const Vec& t_gt);