#include "libOrsa/libNumerics/matrix.h"
#include <vector>
#include "PoseEstimation.h"
#include "Pipeline.h"
#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include "Match.h"
#include <Eigen/Dense>
#include "FNS.h"

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;


using namespace Imagine;

Vec TriangulatePoint(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime){
    double u= U(0);
    double v= U(1);
    double u_p= U_prime(0);
    double v_p= U_prime(1);

    // creating the matrix A that is 4x4
    Mat A= Mat::zeros(4,4);

    Mat p0=P.copyRows(0,0);
    Mat p1=P.copyRows(1,1);
    Mat p2=P.copyRows(2, 2);

    Mat p0p=P_prime.copyRows(0,0);
    Mat p1p=P_prime.copyRows(1,1);
    Mat p2p=P_prime.copyRows(2, 2);

    // Row 0: uP3T-P1T
    Mat row0= u*p2 - p0;
    // Row 1: vP3T-P2T
    Mat row1= v*p2 - p1;
    // Row 2: u'P3'T - P1'T  
    Mat row2= u_p*p2p - p0p;
    // Row 3: v'P3'T - P2'T
    Mat row3= v_p*p2p - p1p;

    A.paste(0, 0, row0);
    A.paste(1,0, row1);
    A.paste(2,0, row2);
    A.paste(3,0, row3);

    // Solving Minimum of AX subject to ||X||=1 using SVD Decomposition
    Mat W(4,4);
    Mat V(4,4);
    Vec S(4);
    EigenSVD(A, W, S, V);

    Vec solution = V.col(V.ncol()-1);

    return solution;
}

// assuming P1= K1[I|0]  P2=K2[R|t]
// U=(u,v,1) and U_prime=(u',v',1) are pixel coordinates 
int Triangulate(Image<Color,2> I1, Image<Color,2> I2, const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime, const Mat& R, const Vec& t){
    // extracting the coordinates from the u and u' vectors
    double u= U(0);
    double v= U(1);
    double u_p= U_prime(0);
    double v_p= U_prime(1);

    // creating the matrix A that is 4x4
    Mat A= Mat::zeros(4,4);

    Mat p0=P.copyRows(0,0);
    Mat p1=P.copyRows(1,1);
    Mat p2=P.copyRows(2, 2);

    Mat p0p=P_prime.copyRows(0,0);
    Mat p1p=P_prime.copyRows(1,1);
    Mat p2p=P_prime.copyRows(2, 2);

    // Row 0: uP3T-P1T
    Mat row0= u*p2 - p0;
    // Row 1: vP3T-P2T
    Mat row1= v*p2 - p1;
    // Row 2: u'P3'T - P1'T  
    Mat row2= u_p*p2p - p0p;
    // Row 3: v'P3'T - P2'T
    Mat row3= v_p*p2p - p1p;

    A.paste(0, 0, row0);
    A.paste(1,0, row1);
    A.paste(2,0, row2);
    A.paste(3,0, row3);

    // Solving Minimum of AX subject to ||X||=1 using SVD Decomposition
    Mat W(4,4);
    Mat V(4,4);
    Vec S(4);
    EigenSVD(A, W, S, V);
    // A.SVD(W,S,V);

    Vec solution = V.col(V.ncol()-1);

    double w_hom = solution(3);

    if(std::abs(w_hom) > 1e-9){
        double X= solution(0) / w_hom;
        double Y= solution(1) / w_hom;
        double Z= solution(2) / w_hom;

        Vec X1(3);
        X1(0)=X;
        X1(1)=Y;
        X1(2)=Z;

        Vec X2=R*X1+t;
        double z1=X1(2);
        double z2=X2(2);

        // std::cout << "z1=" << z1 << " z2=" << z2 << std::endl;

        if(z1<-100 || z2<-100){
            // vector<Match> matches;
            // Match m;
            // m.x1=u;
            // m.y1=v;
            // m.x2=u_p;
            // m.y2=v_p;

            // matches.push_back(m);

            // int W = I1.width() + I2.width();
            // int H = max(I1.height(), I2.height());
            // Window w1 = openWindow(W, H);
            // drawMatches( w1, I1, I2, matches);
            // click();
            // closeWindow(w1);
            // std::cout << "z1=" << z1 << " z2=" << z2 << std::endl;
        }

        if(z1>0 && z2>0){
            return 1;
        }
        else{
            return -1;
        }


    }
    else{
        std::cout << "w_hom was zero" << std::endl;
        return -1; // same res as if point behind camera
    }
}

double reprojError(const Vec& X_3d_homogeneous, const Mat& P, double u_observed, double v_observed) {
    Vec proj = P * X_3d_homogeneous; // 3x1
    double u_proj = proj(0)/proj(2);
    double v_proj = proj(1)/proj(2);
    return std::sqrt((u_proj-u_observed)*(u_proj-u_observed) + (v_proj-v_observed)*(v_proj-v_observed));
}

// function that normalizes a 3x3 matrix by converting it to a 9-vector form and dividing it by qnorm and converting it back to matrix form
Mat Normaliza_Mat(const Mat& A){
    Vec A_vec(9);
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            A_vec((i*3) +j)=A(i,j);
        }
    }
    A_vec/=std::sqrt(A_vec.qnorm());

    Mat A_norm=Mat::zeros(3);

    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            A_norm(i,j)=A_vec(i * 3 + j);;
        }
    }

    return A_norm;

}

// Reorders columns of U, V (and S) so the near-zero singular value ends up at index 2,
// regardless of what order the library's SVD returned them in.
void reorderForEssentialDecomposition(Mat& U, Vec& S, Mat& V) {
    int minIdx = 0;
    for (int i = 1; i < 3; ++i)
        if (S(i) < S(minIdx)) minIdx = i;

    if (minIdx == 2) return; // already in the right spot

    std::vector<int> order(3);
    if (minIdx == 0)      order = {1, 2, 0};  // even permutation
    else /* minIdx==1 */  order = {2, 0, 1};  // even permutation

    Mat U_new(3,3), V_new(3,3);
    Vec S_new(3);
    for (int newCol = 0; newCol < 3; ++newCol) {
        int oldCol = order[newCol];
        for (int r = 0; r < 3; ++r) {
            U_new(r, newCol) = U(r, oldCol);
            V_new(r, newCol) = V(r, oldCol);
        }
        S_new(newCol) = S(oldCol);
    }
    U = U_new; V = V_new; S = S_new;
}

Mat skew(const Vec& t) {
    Mat T = Mat::zeros(3,3);
    T(0,1) = -t(2); T(0,2) =  t(1);
    T(1,0) =  t(2); T(1,2) = -t(0);
    T(2,0) = -t(1); T(2,1) =  t(0);
    return T;
}

double compareE(const Mat& E_est, const Mat& E_gt) {
    // normalize both to unit Frobenius norm
    Vec e1(9), e2(9);
    for(int i=0;i<3;i++) for(int j=0;j<3;j++){
        e1(i*3+j) = E_est(i,j);
        e2(i*3+j) = E_gt(i,j);
    }
    e1 /= std::sqrt(e1.qnorm());
    e2 /= std::sqrt(e2.qnorm());

    double diff_pos = (e1 - e2).qnorm();
    double diff_neg = (e1 + e2).qnorm();

    return std::sqrt(std::min(diff_pos, diff_neg)); // normalized Frobenius distance, sign-invariant
}

// assumes P1=[I|0] and returns P2=[R|t] 
Mat EstimatePose(Image<Color,2> I1, Image<Color,2> I2, const Mat& K1, const Mat& K2, const Mat& F, const std::vector<Point2D>& img1Pts, const std::vector<Point2D>& img2Pts, const Mat& R_gt, const Vec& t_gt){
    Mat E=K2.t()*F*K1;

    // Normalizing E to unit Frobenius norm
    Vec e_vec(9);
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) e_vec(i*3+j) = E(i,j);
    double norm = std::sqrt(e_vec.qnorm());  
    Mat E_norm = E / norm;

    Mat U(E_norm.nrow(), E_norm.nrow());
    Mat V(E_norm.ncol(), E_norm.ncol());
    Vec S(std::min(E_norm.nrow(), E_norm.ncol()));
    EigenSVD3x3(E_norm, U, S, V);
    // E_norm.SVD(U, S, V);

    // debugging:
    // std::cout << "3 singular values of E: " << std::endl;
    // std::cout << S(0) << " " << S(1) <<  " " << S(2) << std::endl;

    // enforcing singular value correction (s,s,0)
    int minIdx = 0;
    for (int i = 1; i < 3; ++i)
        if (S(i) < S(minIdx)) minIdx = i;

    // average the other two 
    double s = 0.0;
    int count = 0;
    for (int i = 0; i < 3; ++i) {
        if (i != minIdx) { s += S(i); count++; }
    }
    s /= count;

    Mat Sigma = Mat::zeros(3);
    for (int i = 0; i < 3; ++i)
        Sigma(i,i) = (i == minIdx) ? 0.0 : s;

    E_norm = U * Sigma * V.t();

    // debugging E_norm and E_gt
    Mat E_gt = skew(t_gt) * R_gt;
    // std::cout << "difference between E and E_gt: " << compareE(E_norm, E_gt) << std::endl;

    EigenSVD3x3(E_norm, U, S, V);
    // E_norm.SVD(U, S, V);

    // reorderForEssentialDecomposition(U, S, V);

    Mat W=Mat::zeros(3);
    W(0,1)=-1;
    W(1,0)=1;
    W(2,2)=1;

    Mat R1=U*W*V.t();
    Mat R2=U*W.t()*V.t();

    // U is orthogonal --> vector t is a unit vector --> only direction no magnitude . R is a full, correctly-scaled rotation, but t is only a unit direction
    Vec t1=U.col(2);
    Vec t2=-t1;

    // enforcing proper rotation
    if(R1.det()<0){
        R1=-R1;
    }

    if(R2.det()<0){
        R2=-R2;
    } 

    // we got the 4 possibilities of poses: R1,t1 - R1, t2 - R2,t1 - R2,t2

    // debugging directly with gt:
    // double best_rot_err = 1e9, best_trans_err = 1e9;
    // Mat candidates_R[4] = {R1, R1, R2, R2};
    // Vec candidates_t[4] = {t1, t2, t1, t2};
    // for (int i = 0; i < 4; i++) {
    //     double re = rotation_error(R_gt, candidates_R[i]);
    //     double te = translation_error(t_gt, candidates_t[i]);
    //     std::cout << "candidate " << i << ": rot_err=" << re << " trans_err=" << te << std::endl;
    // }

    Mat P1=Mat::zeros(3,4);
    P1.paste(0,0,Mat::eye(3));
    P1.paste(0,3,Vec(0,0,0).col(0));

    Mat P2_1=Mat::zeros(3,4);
    P2_1.paste(0, 0, R1.copy(0,2,0,2));
    P2_1.paste(0,3,t1.col(0));

    Mat P2_2=Mat::zeros(3,4);
    P2_2.paste(0, 0, R1.copy(0,2,0,2));
    P2_2.paste(0,3,t2.col(0));

    Mat P2_3=Mat::zeros(3,4);
    P2_3.paste(0, 0, R2.copy(0,2,0,2));
    P2_3.paste(0,3,t1.col(0));

    Mat P2_4=Mat::zeros(3,4);
    P2_4.paste(0, 0, R2.copy(0,2,0,2));
    P2_4.paste(0,3,t2.col(0));


    // cheirality loop:
    // we loop through the point correspondences and for each point we run the triangulation to compute the cheirality with each of the four poses:
    int total_1=0, total_2=0, total_3=0, total_4=0;

    for(size_t i=0; i<img1Pts.size(); i++){
        Vec u(3);
        Vec u_prime(3);

        u(0)=img1Pts[i].x;
        u(1)=img1Pts[i].y;
        u(2)=1.0;

        u_prime(0)=img2Pts[i].x;
        u_prime(1)=img2Pts[i].y;
        u_prime(2)=1.0;

        if (Triangulate(I1, I2, u,u_prime,K1*P1,K2*P2_1, R1, t1) > 0)total_1++;
        if (Triangulate(I1, I2, u,u_prime,K1*P1,K2*P2_2, R1, t2) > 0)total_2++;
        if (Triangulate(I1, I2, u,u_prime,K1*P1,K2*P2_3, R2, t1) > 0)total_3++;
        if (Triangulate(I1, I2, u,u_prime,K1*P1,K2*P2_4, R2, t2) > 0)total_4++;
    }

    Mat P1_pixel = K1 * P1;
    Mat P2_1_pixel = K2 * P2_1;
    Mat P2_2_pixel = K2 * P2_2;
    Mat P2_3_pixel = K2 * P2_3;
    Mat P2_4_pixel = K2 * P2_4;

    double reproj_total[4] = {0.0, 0.0, 0.0, 0.0};
    int reproj_count[4] = {0, 0, 0, 0};

    Mat P2_pixel_arr[4] = {P2_1_pixel, P2_2_pixel, P2_3_pixel, P2_4_pixel};

    for (size_t i = 0; i < img1Pts.size(); i++) {
        Vec u(3), u_prime(3);
        u(0)=img1Pts[i].x; u(1)=img1Pts[i].y; u(2)=1.0;
        u_prime(0)=img2Pts[i].x; u_prime(1)=img2Pts[i].y; u_prime(2)=1.0;

        for (int c = 0; c < 4; c++) {
            Vec X_hom = TriangulatePoint(u, u_prime, P1_pixel, P2_pixel_arr[c]);
            double e1 = reprojError(X_hom, P1_pixel, u(0), u(1));
            double e2 = reprojError(X_hom, P2_pixel_arr[c], u_prime(0), u_prime(1));
            reproj_total[c] += (e1 + e2);
            reproj_count[c]++;
        }
    }

    double reproj_mean[4];
    for (int c = 0; c < 4; c++) {
        reproj_mean[c] = (reproj_count[c] > 0) ? reproj_total[c] / reproj_count[c] : 1e18;
    }

    // std::cout << "Totals: " << std::endl;
    // std::cout << total_1 << " "
    //       << total_2 << " "
    //       << total_3 << " "
    //       << total_4 << std::endl;

    // deciding the winner: cheirality first, reprojection error as tie-breaker
    int totals[4] = {total_1, total_2, total_3, total_4};
    Mat P2_candidates[4] = {P2_1, P2_2, P2_3, P2_4};

    int best = 0;
    for (int c = 1; c < 4; c++) if (totals[c] > totals[best]) best = c;

    // Checking if any other candidate is "close enough" in vote count to be a real contender
    double margin_threshold = 0.2; // HYPERPARAMETER.  within 20% of the winner's vote count
    Mat bestP = P2_candidates[best];
    double bestReprojScore = reproj_mean[best];

    int finalBest = best; // track index explicitly

    for (int c = 0; c < 4; c++) {
        if (c == best) continue;
        if (totals[c] > 0 &&
            (double)(totals[best] - totals[c]) / (double)totals[best] < margin_threshold) {
            // close contender --> we use reprojection error to decide
            if (reproj_mean[c] < bestReprojScore) {
                bestP = P2_candidates[c];
                bestReprojScore = reproj_mean[c];
                finalBest=c;
            }
        }
    }

    // std::cout << "Totals: " << totals[0] << " " << totals[1] << " " << totals[2] << " " << totals[3] << std::endl;
    // std::cout << "Reproj means: " << reproj_mean[0] << " " << reproj_mean[1] << " " << reproj_mean[2] << " " << reproj_mean[3] << std::endl;
    // std::cout << "Winner by votes: " << best << std::endl;
    // std::cout << "Final winner (post-tiebreak): " << finalBest << std::endl;

    return bestP;

    // int maxTotal = total_1;
    // Mat bestP = P2_1;

    // if (total_2 > maxTotal) {
    //     maxTotal = total_2;
    //     bestP = P2_2;
    // }

    // if (total_3 > maxTotal) {
    //     maxTotal = total_3;
    //     bestP = P2_3;
    // }

    // if (total_4 > maxTotal) {
    //     maxTotal = total_4;
    //     bestP = P2_4;
    // }

    // return bestP;

}