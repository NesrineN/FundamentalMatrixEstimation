#pragma once
#include <vector>
#include "../external/libOrsa/libNumerics/matrix.h"
#include "Match.h"

// Returns F in pixel-space (x2^T F x1 = 0)
// matches is filtered in-place to the inlier set, just like computeF
libNumerics::matrix<double> computeF_ORSA(std::vector<SiftMatch>& matches,
                                          int width1, int height1,
                                          int width2, int height2,
                                          double* outSigma = nullptr);