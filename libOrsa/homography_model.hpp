/**
* @file homography_model.hpp
* @brief Homography matrix model
* @author Pascal Monasse, Pierre Moulon
* 
* Copyright (c) 2011,2020-2021 Pascal Monasse
* Copyright (c) 2011 Pierre Moulon
* All rights reserved.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef HOMOGRAPHY_MODEL_H_
#define HOMOGRAPHY_MODEL_H_

#include "model_estimator.hpp"
#include "match.hpp"

namespace orsa {

/// Homography model used for robust estimation with ORSA algorithm.
class HomographyModel : public ModelEstimator {
public:
    HomographyModel(const std::vector<Match> &m,
                    const int width1, const int height1, const int width2,
                    const int height2, bool symmetricError = false);

    /// 4 point correspondences required to compute a homography.
    int SizeSample() const { return 4; }

    /// Only 1 homography can be estimated from a sample of 4 points.
    int NbModels() const { return 1; }

    /// Degree of freedom of the model
    int nDegreeOfFreedom() const {return 8;};

    /// Distance used to distinguish inlier/outlier is to a point
    virtual bool DistToPoint() const { return true; }

    /// Estimated homography satisfies the equation y = H x.
    void Fit(const std::vector<int> &indices, std::vector<Mat> *H) const;

    /// Square reprojection error for a given point through the model H.
    double Error(const Model &H, int index, int *side = 0) const;

    double Error(const Model &H, Mat testData, int *side = 0) const;

    /// Computes the area of the inlier region according to the AC-RANSAC
    /// paper for a given error margin. The inlier region is over-estimated to
    /// ease computation. This is a point-to-point error. Used for LRT.
    /// \param sigma The error margin.
    double pSigma(double sigma, bool leftSide = true) const;

    /// Returns the fundamental matrix. For Homography the output is the input.
    /// Necessary for essential matrix.
    inline Model toPixelSpace(const Model &E) const { return E; }

private:
    Mat N1_, N2_; ///< Normalization matrices
    // Area of both images.
    const int _areaLeft, _areaRight;

    void Unnormalize(Model *model) const;

    bool IsOrientationPreserving(const std::vector<int> &indices,
                                 const Mat &H) const;
};

}  // namespace orsa

#endif
