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

#pragma once

struct Match {
    double x1, y1, x2, y2;
};

using namespace Imagine;
using namespace std;

void algoSIFT(Image<Color,2> I1, Image<Color,2> I2, vector<Match>& matches);

vector<FMatrix<float,3,3>> compute_N(vector<Match>& matches);

vector<Match> normalize_matches(FMatrix<float,3,3> N1, FMatrix<float,3,3> N2, vector<Match>& matches);

vector<int> mark_inliers(FMatrix<float,3,3>& Fcandid, vector<Match>& matches, float distMax);

FMatrix<float,3,3> eightpointalgo(vector<Match>& matches);

FMatrix<float,3,3> computeF(vector<Match>& matches);

vector<Match> GetInliers(const std::string& I1_path, const std::string& I2_path, Mat& F_RANSAC);
