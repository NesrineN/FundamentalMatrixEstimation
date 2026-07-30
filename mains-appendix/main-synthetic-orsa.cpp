#include "libOrsa/libNumerics/matrix.h"
#include "FNS.h"
#include "GaussNewton.h"
#include "Initialization.h"
#include "PoseEstimation.h"
#include "Pipeline.h"
#include "Match.h"
#include "GetInliers.h"
#include "ORSAWrapper.h"

#include <vector>
#include <numeric>  
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <iostream> 
#include <cmath>
#include <algorithm>


#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

using namespace Imagine;
using namespace std;
// assumes Mat, Vec, Point2D, Match, skew(), compareE(), computeF(), GetF(),
// FMatrix<float,3,3>, and all your existing helper functions are available


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

void runSyntheticFPipelineTest() {
    std::cout << "===== SYNTHETIC F-ESTIMATION PIPELINE TEST =====" << std::endl;

    // 1. Known camera intrinsics
    Mat K1 = Mat::eye(3);
    K1(0,0)=500; K1(1,1)=500; K1(0,2)=320; K1(1,2)=240;
    Mat K2 = K1;

    // 2. Known ground-truth relative pose (use the LARGE rotation that passed cleanly)
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

    // 3. Generate random 3D points, project into both images (noiseless)
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> distXY(-1.0, 1.0);
    std::uniform_real_distribution<double> distZ(3.0, 8.0);

    std::vector<SiftMatch> matches; // <-- this is the format computeF/GetF expect

    int N = 200; // use more points than the earlier N=50, since RANSAC/8-point
                 // benefit from a larger sample, and real pipelines see hundreds of matches
    for (int i = 0; i < N; i++) {
        Vec X(4);
        X(0) = distXY(gen);
        X(1) = distXY(gen);
        X(2) = distZ(gen);
        X(3) = 1.0;

        Vec p1 = P1_pixel * X;
        double x1 = p1(0)/p1(2);
        double y1 = p1(1)/p1(2);

        Vec p2 = P2_pixel * X;
        double x2 = p2(0)/p2(2);
        double y2 = p2(1)/p2(2);

        if (x1 < 0 || x1 > 640 || y1 < 0 || y1 > 480) continue;
        if (x2 < 0 || x2 > 640 || y2 < 0 || y2 > 480) continue;

        SiftMatch m;
        m.x1 = x1; m.y1 = y1;
        m.x2 = x2; m.y2 = y2;
        matches.push_back(m);
    }

    std::cout << "Generated " << matches.size() << " valid synthetic matches." << std::endl;

    // 4. Ground-truth E, for comparison
    Mat E_true = skew(t_true) * R_true;

    // 5. Feed these EXACT synthetic correspondences through your REAL F pipeline

    // --- Step A: RANSAC + 8-point (computeF operates on FMatrix<float,3,3> / vector<Match>) ---
    // computeF filters 'matches' in-place down to its inlier set, and returns F in HZ convention
    // (x2^T F x1 = 0), matching your real GetInliers flow.
    vector<SiftMatch> matches_for_ransac = matches; // computeF mutates its input, keep a copy if needed

    vector<FMatrix<double,3,3>> N_list = compute_N(matches_for_ransac); // or whatever your actual signature is
    FMatrix<double,3,3> N1=N_list[0];
    FMatrix<double,3,3> N2=N_list[1];

    Mat N1_mat(3,3);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            N1_mat(i,j) = N1(i,j);

    Mat N2_mat(3,3);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            N2_mat(i,j) = N2(i,j);

    std::cout << "N1:" << std::endl; printM(N1_mat);
    std::cout << "N2:" << std::endl; printM(N2_mat);

    // FMatrix<double,3,3> F_RANSAC_f = computeF(matches_for_ransac);

    double sigma_orsa = 0.0;
    Mat F_RANSAC_f = computeF_ORSA(matches_for_ransac, 640, 480, 640, 480, &sigma_orsa);
    std::cout << "ORSA estimated sigma: " << sigma_orsa << std::endl;

    std::cout << "RANSAC inliers kept: " << matches_for_ransac.size()
               << " / " << matches.size() << std::endl;

    // Convert FMatrix<float,3,3> -> Mat (double) for the rest of the pipeline
    Mat F_RANSAC(3,3);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            F_RANSAC(i,j) = F_RANSAC_f(i,j);

    std::cout << "F_RANSAC (raw from computeF, before FNS):" << std::endl; printM(F_RANSAC);

    // --- Step B: prep for FNS (same convention flip you use in your real pipeline) ---
    // Mat F_RANSAC_for_FNS = F_RANSAC.t(); // FNS solves x^T F x' = 0, RANSAC gives x'^T F x = 0
    Mat F_RANSAC_for_FNS = F_RANSAC; // FNS solves x^T F x' = 0, RANSAC gives x'^T F x = 0

    std::vector<Point2D> img1Pts, img2Pts;
    for (size_t i = 0; i < matches_for_ransac.size(); i++) {
        Point2D p1, p2;
        p1.x = matches_for_ransac[i].x1; p1.y = matches_for_ransac[i].y1;
        p2.x = matches_for_ransac[i].x2; p2.y = matches_for_ransac[i].y2;
        img1Pts.push_back(p1);
        img2Pts.push_back(p2);
    }

    double f0 = 600.0; // Kanatani conditioning constant, same as your real pipeline (image-scale)
    int method = 1;    // FNS

    Mat F_estimated = GetF(img1Pts, img2Pts, F_RANSAC_for_FNS, f0, method);
    // GetF already transposes back to HZ convention (x2^T F x1 = 0) before returning,
    // matching your real pipeline.

    // --- Step C: build E_estimated from F_estimated, exactly like EstimatePose does ---
    Mat E_estimated = K2.t() * F_estimated * K1;


    // testing only E from F_ORSA
    Mat E_from_raw_ORSA_F = K2.t() * F_RANSAC * K1;         // NO transpose
    Mat E_from_transposed_ORSA_F = K2.t() * F_RANSAC.t() * K1;  // WITH transpose (your current FNS-input convention)

    std::cout << "compareE, raw ORSA F: " << compareE(E_from_raw_ORSA_F, E_true) << std::endl;
    std::cout << "compareE, transposed ORSA F: " << compareE(E_from_transposed_ORSA_F, E_true) << std::endl;

    // Normalizing E to unit Frobenius norm
    Vec e_vec(9);
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) e_vec(i*3+j) = E_estimated(i,j);
    double norm = std::sqrt(e_vec.qnorm());  
    E_estimated = E_estimated / norm;

    std::cout << "F_estimated:" << std::endl; printM(F_estimated);
    std::cout << "E_estimated (raw):" << std::endl; printM(E_estimated);
    std::cout << "E_true:" << std::endl; printM(E_true);

    Mat U(3,3), V(3,3);
    Vec S(3);
    EigenSVD3x3(E_estimated, U, S, V);
    // E_estimated.SVD(U, S, V);
    std::cout << "E_estimated singular values: " << S(0) << " " << S(1) << " " << S(2) << std::endl;

    // Verify reconstruction: does U*diag(S)*V^T actually equal E_estimated?
    Mat Sigma_check = Mat::zeros(3);
    Sigma_check(0,0)=S(0); Sigma_check(1,1)=S(1); Sigma_check(2,2)=S(2);
    Mat recon = U * Sigma_check * V.t();
    std::cout << "SVD reconstruction check (should equal E_estimated):" << std::endl; printM(recon);

    // --- Step D: apply the same (s,s,0) singular value correction your real pipeline uses ---
    Mat U2(3,3), V2(3,3);
    Vec S2(3);
    EigenSVD3x3(E_estimated, U2, S2, V2);
    // E_estimated.SVD(U2, S2, V2);

    std::cout << "E_estimated singular values (before correction): "
              << S2(0) << " " << S2(1) << " " << S2(2) << std::endl;

    int minIdx = 0;
    for (int i = 1; i < 3; ++i)
        if (S2(i) < S2(minIdx)) minIdx = i;
    double s = 0.0; int count = 0;
    for (int i = 0; i < 3; ++i) if (i != minIdx) { s += S2(i); count++; }
    s /= count;
    Mat Sigma = Mat::zeros(3);
    for (int i = 0; i < 3; ++i) Sigma(i,i) = (i == minIdx) ? 0.0 : s;
    Mat E_estimated_norm = U2 * Sigma * V2.t();

    // 6. Compare E_estimated (from your REAL pipeline) to E_true
    std::cout << "\ncompareE(E_estimated, E_true) BEFORE (s,s,0) correction: "
              << compareE(E_estimated, E_true) << std::endl;
    std::cout << "compareE(E_estimated_norm, E_true) AFTER (s,s,0) correction: "
              << compareE(E_estimated_norm, E_true) << std::endl;

    // 7. Also report F-level self-consistency (epidistance-style), for completeness
    double avg_epi = 0;
    for (size_t i = 0; i < img1Pts.size(); i++) {
        avg_epi += epidistance(img1Pts[i], img2Pts[i], F_estimated);
    }
    std::cout << "avg epi distance (F_estimated on its own inliers): "
              << avg_epi / img1Pts.size() << std::endl;

    std::cout << "===== END TEST =====" << std::endl;
}

int main() {
    runSyntheticFPipelineTest();
    return 0;
}