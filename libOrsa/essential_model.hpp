/**
* @file essential_model.hpp
* @brief Compute essential matrix
* @author Clement Riu
*
* Copyright (c) 2021 Clement Riu
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

#ifndef MMM_ORSA_ESSENTIAL_MODEL_HPP
#define MMM_ORSA_ESSENTIAL_MODEL_HPP

#include "eigen/Eigen/Eigen"

#include "fundamental_model.hpp"
#include "match.hpp"
#include "model_estimator.hpp"


namespace orsa {

/// Convert an eigen matrix to our own matrix defined in libNumerics/matrix.h
libNumerics::matrix<double> eigenToLibMatrix(const Eigen::MatrixXd &matrixToChange);

/// Apply calibration matrix normalisation to the points to go from pixel points to camera points.
void normalisePoints(const std::vector<Match> &points_,
                     const Eigen::Matrix3d &intrinsicsSrc_,
                     const Eigen::Matrix3d &intrinsicsDst_,
                     std::vector<Match> &normalisedPoints_);

/// Transforms an essential matrix to a fundamental matrix.
inline libNumerics::matrix<double>
essentialToFundamental(const libNumerics::matrix<double> &E, const Eigen::Matrix3d &intrinsicsSrc_,
                       const Eigen::Matrix3d &intrinsicsDst_);

/// Essential matrix model.
class EssentialModel : public FundamentalModel {
public:
    /// Both normalised and un-normalised matches are necessary for computation.
    /// Calibration matrixes are also requiered.
    EssentialModel(const std::vector<Match> &m, const std::vector<Match> &mNormalised,
                   int width1, int height1, int width2, int height2,
                   const Eigen::Matrix3d &intrinsicsSrc_, const Eigen::Matrix3d &intrinsicsDst_,
                   bool symError = false);

    /// 5 point correspondences required to compute an essential matrix.
    int SizeSample() const { return 5; }

    /// 10 Essential matrix can be estimated from a sample of 5 points.
    int NbModels() const { return 10; }

    /// Degree of freedom of the model
    int nDegreeOfFreedom() const { return 5;};

    /// Distance used to distinguish inlier/outlier is to a line
    virtual bool DistToPoint() const { return false; }

    /// Computes a essential matrix given a set of indeces in the matches.
    void Fit(const std::vector<int> &indices, std::vector<Model> *Es) const;

    /// Sampson error in pixel space for a given point through E.
    double Error(const Model &E, int index, int *side = 0) const;

    double Error(const Model &E, Mat testMat, int *side = 0) const;

    /// Computes the fundamental matrix associated with E.
    inline Model toPixelSpace(const Model &E) const;

private:
    /// Essential matrix specific elements are added to the fundamental model.
    const Mat _dataNormalised;
    const Eigen::Matrix3d _intrisicSrc;
    const Eigen::Matrix3d _intrisicDst;

    /// Script extracted from MAGSAC implementation. See essential_model.cpp for reference.
    inline Eigen::Matrix<double, 1, 10> multiplyDegOnePoly(
                                                           const Eigen::RowVector4d &a,
                                                           const Eigen::RowVector4d &b) const;

    /// Script extracted from MAGSAC implementation. See essential_model.cpp for reference.
    inline Eigen::Matrix<double, 1, 20> multiplyDegTwoDegOnePoly(
                                                                 const Eigen::Matrix<double, 1, 10> &a,
                                                                 const Eigen::RowVector4d &b) const;

    /// Script extracted from MAGSAC implementation. See essential_model.cpp for reference.
    inline Eigen::Matrix<double, 10, 20> buildConstraintMatrix(
                                                               const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const;

    /// Script extracted from MAGSAC implementation. See essential_model.cpp for reference.
    inline Eigen::Matrix<double, 9, 20> getTraceConstraint(
                                                           const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const;

    /// Script extracted from MAGSAC implementation. See essential_model.cpp for reference.
    inline Eigen::Matrix<double, 1, 10>
    computeEETranspose(const Eigen::Matrix<double, 1, 4> nullSpace[3][3], int i, int j) const;

    /// Script extracted from MAGSAC implementation. See essential_model.cpp for reference.
    inline Eigen::Matrix<double, 1, 20> getDeterminantConstraint(
                                                                 const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const;
};

} // Namespace Orsa

#endif //MMM_ORSA_ESSENTIAL_MODEL_HPP
