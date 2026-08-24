#include "ORSAWrapper.h"
#include "../external/libOrsa/fundamental_model.hpp"
#include "../external/libOrsa/orsa.hpp"
#include "../external/libOrsa/match.hpp" 

libNumerics::matrix<double> computeF_ORSA(std::vector<SiftMatch>& matches,
                                          int width1, int height1,
                                          int width2, int height2,
                                          double* outSigma) {
    // Convert to libOrsa's Match type
    std::vector<::Match> orsaMatches;  
    orsaMatches.reserve(matches.size());
    for (const auto& m : matches) {
        orsaMatches.emplace_back((float)m.x1, (float)m.y1, (float)m.x2, (float)m.y2);
    }

    bool symError = true;
    orsa::FundamentalModel model(orsaMatches, width1, height1, width2, height2, symError);

    orsa::Orsa algo(&model);
    algo.setHyperParameters();

    orsa::RansacAlgorithm::RunResult res;
    double nfa = algo.run(res, 10000, false);

    libNumerics::matrix<double> F(3,3);
    F.fill(0.0);

    if (!algo.satisfyingRun(nfa)) {
        std::cerr << "ORSA: no satisfying model found" << std::endl;
        matches.clear();
        return F;
    }

    for (int i=0;i<3;i++) for(int j=0;j<3;j++) F(i,j) = res.model(i,j);

    if (outSigma) *outSigma = res.sigma;

    // Filter our matches down to the inlier set, just like computeF
    std::vector<SiftMatch> inlierMatches;
    for (int idx : res.vInliers) inlierMatches.push_back(matches[idx]);
    matches = inlierMatches;

    return F;
}