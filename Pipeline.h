#include "libOrsa/libNumerics/matrix.h"
#include "FNS.h"
#include "GaussNewton.h"
#include "Initialization.h"
#include "./Imagine/Features.h"
#include "PoseEstimation.h"
#include "GetInliers.h"

#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>

const double PI = 3.14159265358979323846;

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

using namespace Imagine;
using namespace std;

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

double epidistance(Point2D p1, Point2D p2, Mat F);

Vec RunPipelineNoiseless(Image<Color,2> I1, Image<Color,2> I2, const std::string& I1_path, const std::string& I2_path, const Mat& K1, const Mat& K2, const double& f0, const Mat& R_gt, const Vec& t_gt, const int& method, double fx, double fy, double cx, double cy,
                       double k1, double k2, double p1, double p2, double k3);