#include <iostream> 
#include <cmath>
#include <vector>
#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include "Match.h"
#include "libOrsa/libNumerics/matrix.h"
#include "PoseEstimation.h"

#pragma once

// struct Match {
//     double x1, y1, x2, y2;
// };

using namespace Imagine;
using namespace std;

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

Point2D undistortPoint(double u, double v,
                        double fx, double fy, double cx, double cy,
                        double k1, double k2, double p1, double p2, double k3);

void undistortMatches(std::vector<SiftMatch>& matches,
                       double fx, double fy, double cx, double cy,
                       double k1, double k2, double p1, double p2, double k3);                        

void algoSIFT(Image<Color,2> I1, Image<Color,2> I2, vector<SiftMatch>& matches);

void removeDuplicateMatches(std::vector<SiftMatch>& matches, double eps = 1e-6);

vector<FMatrix<double,3,3>> compute_N(vector<SiftMatch>& matches);

vector<SiftMatch> normalize_matches(FMatrix<double,3,3> N1, FMatrix<double,3,3> N2, vector<SiftMatch>& matches);

vector<int> mark_inliers(FMatrix<double,3,3>& Fcandid, vector<SiftMatch>& matches, double distMax);

FMatrix<double,3,3> eightpointalgo(vector<SiftMatch>& matches, const FMatrix<double,3,3>& N1, const FMatrix<double,3,3>& N2);

FMatrix<double,3,3> computeF(vector<SiftMatch>& matches);


void drawMatches(Window w, Image<Color,2> I1, Image<Color,2> I2,
                      const vector<SiftMatch>& matches, int maxToDraw = 150);
                      
vector<SiftMatch> GetInliers(const std::string& I1_path, const std::string& I2_path, Mat& F_RANSAC, double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3);