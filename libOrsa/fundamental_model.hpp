/**
 * @file fundamental_model.cpp
 * @brief Fundamental matrix model
 * @author Pascal Monasse, Pierre Moulon
 * 
 * Copyright (c) 2011,2020 Pascal Monasse
 * Copyright (c) 2011 Pierre Moulon
 * All rights reserved.
 *ModelEstimator
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

#ifndef FUNDAMENTAL_MODEL_H_
#define FUNDAMENTAL_MODEL_H_

#include "model_estimator.hpp"
#include "match.hpp"

namespace orsa {

    ///  Fundamental 7-point model, used for robust estimation.
    ///
    /// See page 281 of book by Hartley-Zisserman.
    /// The equation is \f$det(F_1 + \alpha F_2) = 0\f$.
    class FundamentalModel : public ModelEstimator {
    public:
        FundamentalModel(const std::vector<Match> &m, int width1, int height1, int width2, int height2,
                         bool symError = false);

        /// 7 points are required to compute a fundamental matrix.
        int SizeSample() const { return 7; }

        /// Up to 3 fundamental matrices are computed from a sample of 7 points.
        int NbModels() const { return 3; }

        /// Degree of freedom of the model
        int nDegreeOfFreedom() const { return 7; };

        /// Distance used to distinguish inlier/outlier is to a line
        virtual bool DistToPoint() const { return false; }

        void Fit(const std::vector<int> &indices, std::vector<Model> *Fs) const;

        /// Square reprojection error for a given point through F.
        double Error(const Model &F, int index, int *side = 0) const;

        double Error(const Model &F, Mat testMat, int *side = 0) const;

        /// Computes the area of the inlier region according to the AC-RANSAC
        /// paper for a given error margin. The inlier region is over-estimated to
        /// ease computation. This is a point-to-line error. Used for LRT.
        /// \param sigma The error margin.
        double pSigma(double sigma, bool leftSide = true) const;

        /// Returns the fundamental matrix. For Fundamental matrix the output is the input.
        /// Necessary for essential matrix.
        inline Model toPixelSpace(const Model &E) const { return E; }

    protected:
        /// Square reprojection error for a given point through F.
        double
        Error(const Model &F, const double xa, const double xb, const double ya, const double yb, int *side) const;

    private:
        Mat N1_; ///< Normalization for x1_
        Mat N2_; ///< Normalization for x2_
        void Unnormalize(Model *model) const;

        void EpipolarEquation(const std::vector<int> &indices, Mat *A) const;

        void algo7pt(const Mat &A, std::vector<Mat> *Fs,
                     const std::vector<int> &indices) const;

        void algo8pt(const Mat &A, std::vector<Mat> *Fs) const;

        bool checkF(const Mat &F, const std::vector<int> &indices) const;

        // Area and diameter of both images.
        const int _areaLeft, _areaRight;
        const double _diameterLeft, _diameterRight;
    };

}  // namespace orsa

#endif
