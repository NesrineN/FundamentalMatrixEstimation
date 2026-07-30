// ORSAWrapper.h
#pragma once
#include <vector>
#include "./libOrsa/libNumerics/matrix.h"
#include "Match.h"

// Returns F in pixel-space (HZ convention: x2^T F x1 = 0, consistent with your
// existing eightpointalgo convention - VERIFY this against libOrsa's actual
// convention once you test it, since FundamentalModel::Error uses xa=data(0,index)
// i.e. image-1 coords on the "F * x1" side - matches your convention, but confirm
// empirically with a known-good synthetic test as you've done throughout this session).
//
// matches is filtered in-place to the inlier set, mirroring your existing computeF behavior.
libNumerics::matrix<double> computeF_ORSA(std::vector<SiftMatch>& matches,
                                          int width1, int height1,
                                          int width2, int height2,
                                          double* outSigma = nullptr);