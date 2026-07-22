#include "libOrsa/libNumerics/matrix.h"
#include "FNS.h"
#include "GaussNewton.h"
#include "Initialization.h"
#include "PoseEstimation.h"
#include "Pipeline.h"
#include "GetInliers.h"

#include <vector>
#include <random>
#include <iostream>
#include <cmath>

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

using namespace Imagine;
using namespace std;

Mat makeTestRotation(double yaw, double pitch, double roll) {
    Mat Rz = Mat::eye(3);
    Rz(0,0)=cos(yaw); Rz(0,1)=-sin(yaw);
    Rz(1,0)=sin(yaw); Rz(1,1)=cos(yaw);

    Mat Ry = Mat::eye(3);
    Ry(0,0)=cos(pitch); Ry(0,2)=sin(pitch);
    Ry(2,0)=-sin(pitch); Ry(2,2)=cos(pitch);

    Mat Rx = Mat::eye(3);
    Rx(1,1)=cos(roll); Rx(1,2)=-sin(roll);
    Rx(2,1)=sin(roll); Rx(2,2)=cos(roll);

    return Rz * Ry * Rx;
}

// Eigen-backed SVD wrapper, proven reliable on near-degenerate 3x3 matrices
void EigenSVD3x3(const Mat& A, Mat& U, Vec& S, Mat& V) {
    Eigen::Matrix3d eigenA;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) eigenA(i,j)=A(i,j);

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(eigenA, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix3d eU = svd.matrixU();
    Eigen::Matrix3d eV = svd.matrixV();
    Eigen::Vector3d eS = svd.singularValues();

    U = Mat(3,3); V = Mat(3,3); S = Vec(3);
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) { U(i,j)=eU(i,j); V(i,j)=eV(i,j); }
    for (int i=0;i<3;i++) S(i)=eS(i);
}

void runFullSyntheticEndToEndTest() {
    std::cout << "===== FULL SYNTHETIC END-TO-END TEST (real F pipeline + Eigen SVD decomposition) =====" << std::endl;

    // 1. Known intrinsics
    Mat K1 = Mat::eye(3);
    K1(0,0)=500; K1(1,1)=500; K1(0,2)=320; K1(1,2)=240;
    Mat K2 = K1;

    // 2. Known ground truth (large rotation, avoids the small-rotation cheirality fragility)
    Mat R_true = makeTestRotation(0.6, 0.3, 0.2);
    Vec t_true(3);
    t_true(0) = 0.5; t_true(1) = 0.1; t_true(2) = -0.2;

    Mat Eye34 = Mat::zeros(3,4);
    Eye34(0,0)=1; Eye34(1,1)=1; Eye34(2,2)=1;
    Mat P1_pixel = K1 * Eye34;

    Mat P2_34 = Mat::zeros(3,4);
    P2_34.paste(0,0, R_true);
    P2_34.paste(0,3, t_true.col(0));
    Mat P2_pixel = K2 * P2_34;

    // 3. Generate noiseless synthetic correspondences
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> distXY(-1.0, 1.0);
    std::uniform_real_distribution<double> distZ(3.0, 8.0);

    std::vector<Match> matches;
    std::vector<Point2D> img1Pts_all, img2Pts_all;

    int N = 200;
    for (int i = 0; i < N; i++) {
        Vec X(4);
        X(0) = distXY(gen); X(1) = distXY(gen); X(2) = distZ(gen); X(3) = 1.0;

        Vec p1 = P1_pixel * X;
        double x1 = p1(0)/p1(2), y1 = p1(1)/p1(2);

        Vec p2 = P2_pixel * X;
        double x2 = p2(0)/p2(2), y2 = p2(1)/p2(2);

        if (x1 < 0 || x1 > 640 || y1 < 0 || y1 > 480) continue;
        if (x2 < 0 || x2 > 640 || y2 < 0 || y2 > 480) continue;

        Match m; m.x1=x1; m.y1=y1; m.x2=x2; m.y2=y2;
        matches.push_back(m);
    }
    std::cout << "Generated " << matches.size() << " valid synthetic matches." << std::endl;

    // 4. Run REAL F-estimation pipeline: RANSAC + 8-point, then FNS
    vector<Match> matches_for_ransac = matches;
    FMatrix<double,3,3> F_RANSAC_f = computeF(matches_for_ransac);
    std::cout << "RANSAC inliers kept: " << matches_for_ransac.size() << " / " << matches.size() << std::endl;

    Mat F_RANSAC(3,3);
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) F_RANSAC(i,j) = F_RANSAC_f(i,j);

    Mat F_RANSAC_for_FNS = F_RANSAC.t(); // FNS convention flip

    std::vector<Point2D> img1Pts, img2Pts;
    for (size_t i = 0; i < matches_for_ransac.size(); i++) {
        Point2D p1, p2;
        p1.x = matches_for_ransac[i].x1; p1.y = matches_for_ransac[i].y1;
        p2.x = matches_for_ransac[i].x2; p2.y = matches_for_ransac[i].y2;
        img1Pts.push_back(p1);
        img2Pts.push_back(p2);
    }

    double f0 = 600.0;
    int method = 1; // FNS
    Mat F_estimated = GetF(img1Pts, img2Pts, F_RANSAC_for_FNS, f0, method);

    // 5. Build E, normalize, decompose using EigenSVD3x3 (the validated fix)
    Mat E = K2.t() * F_estimated * K1;

    Vec e_vec(9);
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) e_vec(i*3+j) = E(i,j);
    double norm = std::sqrt(e_vec.qnorm());
    Mat E_norm = E / norm;

    Mat U(3,3), V(3,3);
    Vec S(3);
    EigenSVD3x3(E_norm, U, S, V);

    std::cout << "E singular values before (s,s,0) correction: "
              << S(0) << " " << S(1) << " " << S(2) << std::endl;

    int minIdx = 0;
    for (int i = 1; i < 3; ++i) if (S(i) < S(minIdx)) minIdx = i;
    double s = 0.0; int count = 0;
    for (int i = 0; i < 3; ++i) if (i != minIdx) { s += S(i); count++; }
    s /= count;
    Mat Sigma = Mat::zeros(3);
    for (int i = 0; i < 3; ++i) Sigma(i,i) = (i == minIdx) ? 0.0 : s;
    E_norm = U * Sigma * V.t();

    EigenSVD3x3(E_norm, U, S, V); // second SVD, on the corrected E_norm
    // No reorder needed - Eigen returns descending-sorted S and standard (non-transposed) V

    Mat W = Mat::zeros(3);
    W(0,1) = -1; W(1,0) = 1; W(2,2) = 1;

    Mat R1 = U*W*V.t();
    Mat R2 = U*W.t()*V.t();
    Vec t1 = U.col(2);
    Vec t2 = -t1;

    if (R1.det() < 0) R1 = -R1;
    if (R2.det() < 0) R2 = -R2;

    Mat candidates_R[4] = {R1, R1, R2, R2};
    Vec candidates_t[4] = {t1, t2, t1, t2};

    // 6. Compare all 4 candidates directly against ground truth
    std::cout << "\n--- Candidate errors vs ground truth ---" << std::endl;
    for (int i = 0; i < 4; i++) {
        double re = rotation_error(R_true, candidates_R[i]);
        double te = translation_error(t_true, candidates_t[i]);
        std::cout << "candidate " << i << ": rot_err=" << re << " deg, trans_err=" << te << " deg" << std::endl;
    }

    // 7. Cheirality vote on the RANSAC-inlier synthetic points (noiseless)
    int totals[4] = {0,0,0,0};
    for (size_t i = 0; i < img1Pts.size(); i++) {
        Vec u(3), u_p(3);
        u(0)=img1Pts[i].x; u(1)=img1Pts[i].y; u(2)=1.0;
        u_p(0)=img2Pts[i].x; u_p(1)=img2Pts[i].y; u_p(2)=1.0;

        for (int c = 0; c < 4; c++) {
            Mat P2c_34 = Mat::zeros(3,4);
            P2c_34.paste(0,0, candidates_R[c]);
            P2c_34.paste(0,3, candidates_t[c].col(0));
            Mat P2c_pixel = K2 * P2c_34;

            Image<Color,2> dummyI1, dummyI2; // Triangulate's image params are unused/dead-code
            if (Triangulate(dummyI1, dummyI2, u, u_p, P1_pixel, P2c_pixel,
                             candidates_R[c], candidates_t[c]) > 0) {
                totals[c]++;
            }
        }
    }
    std::cout << "\nCheirality totals: " << totals[0] << " " << totals[1] << " "
              << totals[2] << " " << totals[3] << std::endl;

    std::cout << "===== END TEST =====" << std::endl;
}

int main() {
    runFullSyntheticEndToEndTest();
    return 0;
}