#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>

#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

using namespace Imagine;
using namespace std;

struct Match {
    float x1, y1, x2, y2;
};

void algoSIFT(Image<Color,2> I1, Image<Color,2> I2, vector<Match>& matches);

vector<FMatrix<float,3,3>> compute_N(vector<Match>& matches);

vector<Match> normalize_matches(FMatrix<float,3,3> N1, FMatrix<float,3,3> N2, vector<Match>& matches);

vector<int> mark_inliers(FMatrix<float,3,3>& Fcandid, vector<Match>& matches, float distMax);

FMatrix<float,3,3> eightpointalgo(vector<Match>& matches);

FMatrix<float,3,3> computeF(vector<Match>& matches);

void displayEpipolar(Image<Color> I1, Image<Color> I2, const FMatrix<float,3,3>& F);