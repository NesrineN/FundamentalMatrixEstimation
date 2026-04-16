/** Copyright (C) 2019 Czech Technical University.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are
* met:
*
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*
*     * Redistributions in binary form must reproduce the above
*       copyright notice, this list of conditions and the following
*       disclaimer in the documentation and/or other materials provided
*       with the distribution.
*
*     * Neither the name of Czech Technical University nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*
* Please contact the author of this library if you have any questions.
* Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
*/
// Adaptation by Clemen Riu (2021)

#include "essential_model.hpp"

namespace orsa {

/// Convert an eigen matrix to our own matrix defined in libNumerics/matrix.h
libNumerics::matrix<double> eigenToLibMatrix(const Eigen::MatrixXd &matrixToChange) {
    int nRow = matrixToChange.rows();
    int nCol = matrixToChange.cols();
    libNumerics::matrix<double> matrixToReturn(nRow, nCol);
    for (int iRow = 0; iRow < nRow; iRow++) {
        for (int iCol = 0; iCol < nCol; iCol++) {
            matrixToReturn(iRow, iCol) = matrixToChange(iRow, iCol);
        }
    }
    return matrixToReturn;
}

/// Apply calibration matrix normalisation to the points to go from pixel points to camera points.
void normalisePoints(const std::vector<Match> &points_,
                     const Eigen::Matrix3d &intrinsicsSrc_,
                     const Eigen::Matrix3d &intrinsicsDst_,
                     std::vector<Match> &normalisedPoints_) {

    const Eigen::Matrix3d inverse_intrinsics_src = intrinsicsSrc_.inverse(),
        inverse_intrinsics_dst = intrinsicsDst_.inverse();

    // Most likely, this is not the fastest solution, but it does
    // not affect the speed of Graph-cut RANSAC, so not a crucial part of
    // this example.
    double x0, y0, x1, y1;
    std::vector<Match>::const_iterator itPoints = points_.begin();
    for (; itPoints != points_.end(); ++itPoints) {
        Eigen::Vector3d point_src,
            point_dst,
            normalized_point_src,
            normalized_point_dst;

        x0 = (*itPoints).x1;
        y0 = (*itPoints).y1;
        x1 = (*itPoints).x2;
        y1 = (*itPoints).y2;

        point_src << x0, y0, 1.0; // Homogeneous point in the first image
        point_dst << x1, y1, 1.0; // Homogeneous point in the second image

        // Normalized homogeneous point in the first image
        normalized_point_src =
            inverse_intrinsics_src * point_src;
        // Normalized homogeneous point in the second image
        normalized_point_dst =
            inverse_intrinsics_dst * point_dst;

        Match normalisedMatch(normalized_point_src(0), normalized_point_src(1), normalized_point_dst(0),
                              normalized_point_dst(1));
        // The second four columns contain the normalized coordinates.
        normalisedPoints_.push_back(normalisedMatch);
    }
}

/// Transforms an essential matrix to a fundamental matrix.
inline libNumerics::matrix<double>
essentialToFundamental(const libNumerics::matrix<double> &E, const Eigen::Matrix3d &intrinsicsSrc,
                       const Eigen::Matrix3d &intrinsicsDst) {
    return orsa::eigenToLibMatrix(intrinsicsSrc.transpose().inverse()) * E *
        orsa::eigenToLibMatrix(intrinsicsDst.inverse());
}


/// Constructor
EssentialModel::EssentialModel(const std::vector<Match> &m, const std::vector<Match> &mNormalised,
                               int width1, int height1, int width2, int height2,
                               const Eigen::Matrix3d &intrinsicsSrc_, const Eigen::Matrix3d &intrinsicsDst_,
                               bool symError) : FundamentalModel(m, width1, height1,  width2, height2, symError),
                                                _dataNormalised(Match::toMat(mNormalised)),
                                                _intrisicSrc(intrinsicsSrc_), _intrisicDst(intrinsicsDst_) {
}

/// Script extracted from MAGSAC implementation.
/// Computes a essential matrix given a set of indeces in the matches.
void EssentialModel::Fit(const std::vector<int> &indices, std::vector<Model> *Es) const {

    Eigen::MatrixXd coefficients(indices.size(), 9);

    // Step 1. Create the nx9 matrix containing epipolar constraints.
    //   Essential matrix is a linear combination of the 4 vectors spanning the null space of this
    //   matrix.
    double x0, y0, x1, y1, weight = 1.0;
    for (size_t i = 0; i < indices.size(); i++) {
        int index = indices[i];
        x0 = _dataNormalised(0, index);
        y0 = _dataNormalised(1, index);
        x1 = _dataNormalised(2, index);
        y1 = _dataNormalised(3, index);

        // Precalculate these values to avoid calculating them multiple times
        const double
            weight_times_x0 = weight * x0,
            weight_times_x1 = weight * x1,
            weight_times_y0 = weight * y0,
            weight_times_y1 = weight * y1;

        coefficients.row(i) <<
            weight_times_x0 * x1,
            weight_times_x0 * y1,
            weight_times_x0,
            weight_times_y0 * x1,
            weight_times_y0 * y1,
            weight_times_y0,
            weight_times_x1,
            weight_times_y1,
            weight;
    }

    // Extract the null space from a minimal sampling (using LU) or non-minimal sampling (using SVD).
    Eigen::Matrix<double, 9, 4> nullSpace;

    if (indices.size() == 5) {
        const Eigen::FullPivLU<Eigen::MatrixXd> lu(coefficients);
        if (lu.dimensionOfKernel() != 4) {
            return;
        }
        nullSpace = lu.kernel();
    } else {
        const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                                                    coefficients.transpose() * coefficients, Eigen::ComputeFullV);
        nullSpace = svd.matrixV().rightCols<4>();
    }

    const Eigen::Matrix<double, 1, 4> nullSpaceMatrix[3][3] = {
        {nullSpace.row(0), nullSpace.row(3), nullSpace.row(6)},
        {nullSpace.row(1), nullSpace.row(4), nullSpace.row(7)},
        {nullSpace.row(2), nullSpace.row(5), nullSpace.row(8)}};

    // Step 2. Expansion of the epipolar constraints on the determinant and trace.
    const Eigen::Matrix<double, 10, 20> constraintMatrix = buildConstraintMatrix(nullSpaceMatrix);

    // Step 3. Eliminate part of the matrix to isolate polynomials in z.
    Eigen::FullPivLU<Eigen::Matrix<double, 10, 10>> c_lu(constraintMatrix.block<10, 10>(0, 0));
    const Eigen::Matrix<double, 10, 10> eliminatedMatrix = c_lu.solve(constraintMatrix.block<10, 10>(0, 10));


    Eigen::Matrix<double, 10, 10> actionMatrix = Eigen::Matrix<double, 10, 10>::Zero();
    actionMatrix.block<3, 10>(0, 0) = eliminatedMatrix.block<3, 10>(0, 0);
    actionMatrix.row(3) = eliminatedMatrix.row(4);
    actionMatrix.row(4) = eliminatedMatrix.row(5);
    actionMatrix.row(5) = eliminatedMatrix.row(7);
    actionMatrix(6, 0) = -1.0;
    actionMatrix(7, 1) = -1.0;
    actionMatrix(8, 3) = -1.0;
    actionMatrix(9, 6) = -1.0;
    Eigen::EigenSolver<Eigen::Matrix<double, 10, 10>> eigensolver(actionMatrix);
    const Eigen::VectorXcd &eigenvalues = eigensolver.eigenvalues();

    // Now that we have x, y, and z we need to substitute them back into the null space to get a valid
    // essential matrix solution.
    for (size_t i = 0; i < 10; i++) {
        // Only consider real solutions.
        if (eigenvalues(i).imag() != 0) {
            continue;
        }

        Eigen::Matrix3d E_dst_src;
        Eigen::Map<Eigen::Matrix<double, 9, 1>>(E_dst_src.data()) =
            nullSpace * eigensolver.eigenvectors().col(i).tail<4>().real();

        Eigen::MatrixXd model;
        model = E_dst_src;
        Es->push_back(eigenToLibMatrix(model));
    }

    return;
}

/// Script extracted from MAGSAC implementation.
// Multiply two degree one polynomials of variables x, y, z.
// E.g. p1 = a[0]x + a[1]y + a[2]z + a[3]
// Output order: x^2 xy y^2 xz yz z^2 x y z 1 (GrevLex)
inline Eigen::Matrix<double, 1, 10> EssentialModel::multiplyDegOnePoly(
                                                                       const Eigen::RowVector4d &a,
                                                                       const Eigen::RowVector4d &b) const {
    Eigen::Matrix<double, 1, 10> output;
    // x^2
    output(0) = a(0) * b(0);
    // xy
    output(1) = a(0) * b(1) + a(1) * b(0);
    // y^2
    output(2) = a(1) * b(1);
    // xz
    output(3) = a(0) * b(2) + a(2) * b(0);
    // yz
    output(4) = a(1) * b(2) + a(2) * b(1);
    // z^2
    output(5) = a(2) * b(2);
    // x
    output(6) = a(0) * b(3) + a(3) * b(0);
    // y
    output(7) = a(1) * b(3) + a(3) * b(1);
    // z
    output(8) = a(2) * b(3) + a(3) * b(2);
    // 1
    output(9) = a(3) * b(3);
    return output;
}

/// Script extracted from MAGSAC implementation.
// Multiply a 2 deg poly (in x, y, z) and a one deg poly in GrevLex order.
// x^3 x^2y xy^2 y^3 x^2z xyz y^2z xz^2 yz^2 z^3 x^2 xy y^2 xz yz z^2 x y z 1
inline Eigen::Matrix<double, 1, 20> EssentialModel::multiplyDegTwoDegOnePoly(
                                                                             const Eigen::Matrix<double, 1, 10> &a,
                                                                             const Eigen::RowVector4d &b) const {
    Eigen::Matrix<double, 1, 20> output;
    // x^3
    output(0) = a(0) * b(0);
    // x^2y
    output(1) = a(0) * b(1) + a(1) * b(0);
    // xy^2
    output(2) = a(1) * b(1) + a(2) * b(0);
    // y^3
    output(3) = a(2) * b(1);
    // x^2z
    output(4) = a(0) * b(2) + a(3) * b(0);
    // xyz
    output(5) = a(1) * b(2) + a(3) * b(1) + a(4) * b(0);
    // y^2z
    output(6) = a(2) * b(2) + a(4) * b(1);
    // xz^2
    output(7) = a(3) * b(2) + a(5) * b(0);
    // yz^2
    output(8) = a(4) * b(2) + a(5) * b(1);
    // z^3
    output(9) = a(5) * b(2);
    // x^2
    output(10) = a(0) * b(3) + a(6) * b(0);
    // xy
    output(11) = a(1) * b(3) + a(6) * b(1) + a(7) * b(0);
    // y^2
    output(12) = a(2) * b(3) + a(7) * b(1);
    // xz
    output(13) = a(3) * b(3) + a(6) * b(2) + a(8) * b(0);
    // yz
    output(14) = a(4) * b(3) + a(7) * b(2) + a(8) * b(1);
    // z^2
    output(15) = a(5) * b(3) + a(8) * b(2);
    // x
    output(16) = a(6) * b(3) + a(9) * b(0);
    // y
    output(17) = a(7) * b(3) + a(9) * b(1);
    // z
    output(18) = a(8) * b(3) + a(9) * b(2);
    // 1
    output(19) = a(9) * b(3);
    return output;
}


/// Script extracted from MAGSAC implementation.
// Shorthand for multiplying the Essential matrix with its transpose.
inline Eigen::Matrix<double, 1, 10> EssentialModel::computeEETranspose(
                                                                       const Eigen::Matrix<double, 1, 4> nullSpace[3][3],
                                                                       int i,
                                                                       int j) const {
    return multiplyDegOnePoly(nullSpace[i][0], nullSpace[j][0]) +
        multiplyDegOnePoly(nullSpace[i][1], nullSpace[j][1]) +
        multiplyDegOnePoly(nullSpace[i][2], nullSpace[j][2]);
}

/// Script extracted from MAGSAC implementation.
// Builds the trace constraint: EEtE - 1/2 trace(EEt)E = 0
inline Eigen::Matrix<double, 9, 20> EssentialModel::getTraceConstraint(
                                                                       const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const {
    Eigen::Matrix<double, 9, 20> traceConstraint;

    // Compute EEt.
    Eigen::Matrix<double, 1, 10> eet[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            eet[i][j] = 2 * computeEETranspose(nullSpace, i, j);
        }
    }

    // Compute the trace.
    const Eigen::Matrix<double, 1, 10> trace = eet[0][0] + eet[1][1] + eet[2][2];

    // Multiply EEt with E.
    for (auto i = 0; i < 3; i++) {
        for (auto j = 0; j < 3; j++) {
            traceConstraint.row(3 * i + j) = multiplyDegTwoDegOnePoly(eet[i][0], nullSpace[0][j]) +
                multiplyDegTwoDegOnePoly(eet[i][1], nullSpace[1][j]) +
                multiplyDegTwoDegOnePoly(eet[i][2], nullSpace[2][j]) -
                0.5 * multiplyDegTwoDegOnePoly(trace, nullSpace[i][j]);
        }
    }

    return traceConstraint;
}

/// Script extracted from MAGSAC implementation.
inline Eigen::Matrix<double, 10, 20> EssentialModel::buildConstraintMatrix(
                                                                           const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const {
    Eigen::Matrix<double, 10, 20> constraintMatrix;
    constraintMatrix.block<9, 20>(0, 0) = getTraceConstraint(nullSpace);
    constraintMatrix.row(9) = getDeterminantConstraint(nullSpace);
    return constraintMatrix;
}

/// Script extracted from MAGSAC implementation.
inline Eigen::Matrix<double, 1, 20> EssentialModel::getDeterminantConstraint(
                                                                             const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const {
    // Singularity constraint.
    return multiplyDegTwoDegOnePoly(
                                    multiplyDegOnePoly(nullSpace[0][1], nullSpace[1][2]) -
                                    multiplyDegOnePoly(nullSpace[0][2], nullSpace[1][1]),
                                    nullSpace[2][0]) +
        multiplyDegTwoDegOnePoly(
                                 multiplyDegOnePoly(nullSpace[0][2], nullSpace[1][0]) -
                                 multiplyDegOnePoly(nullSpace[0][0], nullSpace[1][2]),
                                 nullSpace[2][1]) +
        multiplyDegTwoDegOnePoly(
                                 multiplyDegOnePoly(nullSpace[0][0], nullSpace[1][1]) -
                                 multiplyDegOnePoly(nullSpace[0][1], nullSpace[1][0]),
                                 nullSpace[2][2]);
}

/// Computes the fundamental matrix associated with E.
inline EssentialModel::Model EssentialModel::toPixelSpace(const Model &E) const {
    return essentialToFundamental(E, _intrisicSrc, _intrisicDst).t();
}

/// Sampson error in pixel space for a given point through E.
/// BEWARE : this takes the fundamental matrix as input.
/// \param F The fundamental matrix.
/// \param index The point correspondence.
/// \param side In which image is the error measured?
/// \return The square reprojection error.
double EssentialModel::Error(const Model &F, int index, int *side) const {
    double xa = data_(0, index), ya = data_(1, index);
    double xb = data_(2, index), yb = data_(3, index);

    return this->FundamentalModel::Error(F, xa, xb, ya, yb, side);
}

/// Sampson error in pixel space for a given point through E.
/// BEWARE : this takes the essential matrix as input.
/// \param E The essential matrix.
/// \param testMat The point correspondence.
/// \param side In which image is the error measured?
/// \return The square reprojection error.
double EssentialModel::Error(const Model &E, Mat testMat, int *side) const {
    const Model F = toPixelSpace(E);
    double xa = testMat(0, 0), ya = testMat(1, 0);
    double xb = testMat(2, 0), yb = testMat(3, 0);

    return this->FundamentalModel::Error(F, xa, xb, ya, yb, side);
}


}  // namespace orsa
