#include "libOrsa/libNumerics/matrix.h"
#include "FNS.h"
#include "GaussNewton.h"
#include "Initialization.h"
#include "./Imagine/Features.h"
#include "PoseEstimation.h"
#include "GetInliers.h"
#include "Match.h"

#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>

const double PI = 3.14159265358979323846;

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

using namespace Imagine;
using namespace std;


// function that creates the vector E from two point-correspondences:
Vec fillE(const Point2D& p1, const Point2D& p2, double f0){
    double x=p1.x;
    double y=p1.y;

    double xp=p2.x;
    double yp=p2.y;

    Vec E(9);

    E(0)=x*xp;
    E(1)=x*yp;
    E(2)=x*f0;
    E(3)=y*xp;
    E(4)=y*yp;
    E(5)=y*f0;
    E(6)=f0*xp;
    E(7)=f0*yp;
    E(8)= f0*f0;

    return E;
}

// Mat computeV0(double x, double y, double xp, double yp, double f0) {
//     // Vectors representing partial derivatives of E with respect to x, y, xp, yp
//     Vec dx(9), dy(9), dxp(9), dyp(9);

//     // dE/dx
//     dx(0) = xp; dx(1) = yp; dx(2) = f0;
//     dx(3) = 0;  dx(4) = 0;  dx(5) = 0;
//     dx(6) = 0;  dx(7) = 0;  dx(8) = 0;

//     // dE/dy
//     dy(0) = 0;  dy(1) = 0;  dy(2) = 0;
//     dy(3) = xp; dy(4) = yp; dy(5) = f0;
//     dy(6) = 0;  dy(7) = 0;  dy(8) = 0;

//     // dE/dxp
//     dxp(0) = x;  dxp(1) = 0;  dxp(2) = 0;
//     dxp(3) = y;  dxp(4) = 0;  dxp(5) = 0;
//     dxp(6) = f0; dxp(7) = 0;  dxp(8) = 0;

//     // dE/dyp
//     dyp(0) = 0;  dyp(1) = x;  dyp(2) = 0;
//     dyp(3) = 0;  dyp(4) = y;  dyp(5) = 0;
//     dyp(6) = 0;  dyp(7) = f0; dyp(8) = 0;

//     // V0 = dx*dx^T + dy*dy^T + dxp*dxp^T + dyp*dyp^T
//     Mat V0 = dx * dx.t() + dy * dy.t() + dxp * dxp.t() + dyp * dyp.t();

//     return V0;
// }

Mat computeV0(double x, double y, double xp, double yp, double f0){

    Mat V0=Mat::zeros(9);
    double s = f0;
    
    // R0
    V0(0,0)= (x*x) + (xp*xp);
    V0(0,1)= xp*yp;
    V0(0,2)= f0 * xp;
    V0(0,3)= x*y;
    V0(0,4)= 0;
    V0(0,5)=0;
    V0(0,6)=f0*x;
    V0(0,7)=0;
    V0(0,8)=0;

    // R1
    V0(1,0)= xp*yp;
    V0(1,1)= x*x + yp*yp;
    V0(1,2)= f0 * yp;
    V0(1,3)= 0;
    V0(1,4)= x*y;
    V0(1,5)=0;
    V0(1,6)=0;
    V0(1,7)=f0*x;
    V0(1,8)=0;

    // R2
    V0(2,0)= f0*xp;
    V0(2,1)= f0*yp;
    V0(2,2)= f0 * f0;
    V0(2,3)= 0;
    V0(2,4)= 0;
    V0(2,5)=0;
    V0(2,6)=0;
    V0(2,7)=0;
    V0(2,8)=0;

    // R3
    V0(3,0)= x*y;
    V0(3,1)= 0;
    V0(3,2)= 0;
    V0(3,3)= y*y + xp*xp;
    V0(3,4)= xp*yp;
    V0(3,5)= f0*xp;
    V0(3,6)= f0*y;
    V0(3,7)=0;
    V0(3,8)=0;

    // R4
    V0(4,0)= 0;
    V0(4,1)= x*y;
    V0(4,2)= 0;
    V0(4,3)= xp*yp;
    V0(4,4)= y*y + yp*yp;
    V0(4,5)= f0*yp;
    V0(4,6)= 0;
    V0(4,7)=f0*y;
    V0(4,8)=0;

    // R5
    V0(5,0)= 0;
    V0(5,1)= 0;
    V0(5,2)= 0;
    V0(5,3)= f0*xp;
    V0(5,4)= f0*yp;
    V0(5,5)= f0*f0;
    V0(5,6)= 0;
    V0(5,7)=0;
    V0(5,8)=0;

    // R6
    V0(6,0)= f0*x;
    V0(6,1)= 0;
    V0(6,2)= 0;
    V0(6,3)= f0*y;
    V0(6,4)= 0;
    V0(6,5)= 0;
    V0(6,6)= f0*f0;
    V0(6,7)=0;
    V0(6,8)=0;

    // R7
    V0(7,0)= 0;
    V0(7,1)= f0*x;
    V0(7,2)= 0;
    V0(7,3)= 0;
    V0(7,4)= f0*y;
    V0(7,5)= 0;
    V0(7,6)= 0;
    V0(7,7)= f0*f0;
    V0(7,8)=0;

    // R8
    V0(8,0)= 0;
    V0(8,1)= 0;
    V0(8,2)= 0;
    V0(8,3)= 0;
    V0(8,4)= 0;
    V0(8,5)= 0;
    V0(8,6)= 0;
    V0(8,7)= 0;
    V0(8,8)=0;

    return V0;
}

Mat GetF(const std::vector<Point2D>& img1Pts, const std::vector<Point2D>& img2Pts, const Mat& F_RANSAC, const double& f0, const int& method){
    Mat Eall=Mat::zeros(9,img1Pts.size());
    std::vector<Mat> Vall;

    for(int i=0; i<img1Pts.size(); i++){
        Point2D p1=img1Pts[i];
        Point2D p2=img2Pts[i];

        Vec E=fillE(p1,p2,f0);

        for(int j = 0; j < 9; ++j)
        {
            Eall(j, i) = E(j);
        }

        double x=p1.x;
        double y=p1.y;
        double xp=p2.x;
        double yp=p2.y;

        Mat V0=computeV0(x,y,xp,yp,f0);

        // adding V0 to the list Vall
        Vall.push_back(V0);
    }

    // we initialize uinit using Taubin method
    // Vec uinit= Taubin(Eall, f0, Vall);

    // Mat F=Mat::zeros(3);
    // for(int i=0; i<3; i++){
    //     for(int j=0; j<3; j++){
    //         F(i, j) = uinit(i * 3 + j); 
    //     }
    // }

    // ADDED AFTER ORSA:
    // Vec f_vec(9);
    // for (int i=0;i<3;i++) for (int j=0;j<3;j++) f_vec(i*3+j) = F_RANSAC(i,j);
    // double f_norm = std::sqrt(f_vec.qnorm());
    // Mat F_RANSAC_normalized = F_RANSAC / f_norm;

    // we initialize uinit using RANSAC's F after integrating /f0 into it: 
    Mat F_kan_init = F_RANSAC;  
    F_kan_init(0,2) /= f0;
    F_kan_init(1,2) /= f0;
    F_kan_init(2,0) /= f0;
    F_kan_init(2,1) /= f0;
    F_kan_init(2,2) /= (f0*f0);

    Vec uinit(9);
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            uinit(i*3 + j)=F_kan_init(i,j);
        }
    }
    uinit/=std::sqrt(uinit.qnorm());

    Mat F=Mat::zeros(3);

    if(method==1){
        // we run FNS to get F
        F =FNS(uinit, Eall, Vall);
    }
    else{
        F = GaussNewton(uinit, Eall, Vall);
    }
    
    F=F.t(); // transposing back to get the F that solves for x'T F x=0 -- the standard Hartley & Zisserman convention 

    // REMOVED THIS IN ORSA
    // De-normalizing F:
    Mat Norm=Mat::eye(3);
    Norm(2,2)=f0;
    Mat F_denorm=Norm.t()*F*Norm;

    return F_denorm;
}

void printM(const Mat& A){
    std::cout << "Matrix: " << std::endl;
    for(int i=0;i<A.nrow(); i++){
        for(int j =0; j<A.ncol(); j++){
            std::cout << A(i,j) << " "; 
        }
        std::cout << "" << std::endl;
        std::cout << "" << std::endl;
    }
}

void printV(const Vec& V){
    std::cout << "Vector: " << std::endl;
    for(int i=0;i<V.nrow(); i++){
        std::cout << V(i) << std::endl;
    }

}

double trace(const Mat& R)
{
    return R(0,0) + R(1,1) + R(2,2);
}

inline double clamp(double x, double a, double b)
{
    return std::max(a, std::min(x, b));
}

double norm(const Vec& v)
{
    return std::sqrt(
        v(0)*v(0) +
        v(1)*v(1) +
        v(2)*v(2));
}


double rotation_error(const Mat& R_gt, const Mat& R_pred){
    Mat R_diff = R_gt.t() * R_pred;

    double c =(trace(R_diff) - 1.0) * 0.5;

    c = clamp(c, -1.0, 1.0);

    double angle =std::acos(c);

    return angle * 180.0 / PI;
}

double translation_error(const Vec& t_gt, const Vec& t_pred){
    
    double n1 = norm(t_gt);
    double n2 = norm(t_pred);

    if (n1 < 1e-12 || n2 < 1e-12)
        return 180.0;

    Vec a = t_gt;
    Vec b = t_pred;

    // normalizing the vectors 
    a /= n1;
    b /= n2;

    double c = (dot(a, b));

    c = clamp(c, -1.0, 1.0);

    double angle = std::acos(c);

    return angle * 180.0 / PI;
}

// returns the distance from x' to the epipolar line Fx
double epidistance(Point2D p1, Point2D p2, Mat F){
    double x=p1.x;
    double y=p1.y;
    double xp=p2.x;
    double yp=p2.y;

    Mat X=Mat::zeros(3,1);
    Mat Xp=Mat::zeros(3,1);
    X(0,0)=x;
    X(1,0)=y;
    X(2,0)=1.0;

    Xp(0,0)=xp;
    Xp(1,0)=yp;
    Xp(2,0)=1.0;


    Mat l=F*X; // l is a 3x1 mat

    double num=std::abs(l(0,0)*Xp(0,0) + l(1,0)*Xp(1,0) + l(2,0));
    double denom=std::sqrt((l(0,0)*l(0,0)) + (l(1,0)*l(1,0)));
    if(denom<1e-6)denom+=1e-6;
    
    return num/denom;
}


// Computes the fraction of an N x N grid over the image that contains at least
// one match point (in image 1). Returns a value in [0,1]; higher = more spread out.
double computeGridCoverage(const vector<SiftMatch>& matches, int imgWidth, int imgHeight, int gridN = 8) {
    vector<vector<bool>> occupied(gridN, vector<bool>(gridN, false));

    double cellW = (double)imgWidth / gridN;
    double cellH = (double)imgHeight / gridN;

    for (const auto& m : matches) {
        int cx = std::min(gridN - 1, (int)(m.x1 / cellW));
        int cy = std::min(gridN - 1, (int)(m.y1 / cellH));
        if (cx >= 0 && cy >= 0) occupied[cy][cx] = true;
    }

    int occupiedCount = 0;
    for (int i = 0; i < gridN; i++)
        for (int j = 0; j < gridN; j++)
            if (occupied[i][j]) occupiedCount++;

    return (double)occupiedCount / (gridN * gridN);
}


Vec RunPipelineNoiseless(Image<Color,2> I1, Image<Color,2> I2, const std::string& I1_path, const std::string& I2_path, const Mat& K1, const Mat& K2, const double& f0, const Mat& R_gt, const Vec& t_gt, const int& method, double fx, double fy, double cx, double cy,
                       double k1, double k2, double p1, double p2, double k3){

    // 1. Get the matches/inliers from the image pairs usi`ng SIFT + RANSAC and 8-point algorithm as a starting point
    Mat F_RANSAC=Mat::zeros(3);
    vector<SiftMatch> matches=GetInliers(I1_path, I2_path, F_RANSAC, fx,  fy,  cx,  cy,  k1,  k2,  p1,  p2,  k3);
    
    // removed for ORSA
    // F_RANSAC=F_RANSAC.t(); // transposing before giving it to FNS as initial F estimation because FNS is solving xT F x' =0 and RANSAC 8-point algo was solving x'T F x=0

    // 2. Using the matches, we want to fill the vectors of 2D point matches img1Pts and img2Pts
    std::vector<Point2D> img1Pts;
    std::vector<Point2D> img2Pts;

    for(int i=0; i<matches.size(); i++){
        Point2D p1;
        Point2D p2;

        p1.x=matches[i].x1;
        p1.y=matches[i].y1;

        p2.x=matches[i].x2;
        p2.y=matches[i].y2;

        img1Pts.push_back(p1);
        img2Pts.push_back(p2);
    }

    // 3. Now, we want to run the method of choice: either FNS or Gauss-Newton to compute the fundamental matrix from the inliers
    if(method==1){
        // std::cout << "----Computing F using FNS----" << std::endl;
    }
    else{
        // std::cout << "----Computing F using Gauss-Newton----" << std::endl;
    }

    Mat F=GetF(img1Pts, img2Pts, F_RANSAC, f0, method);

    // debugging. making sure the epipolar error is small:

    double avg_epidist_estim=0;
    double avg_epidist_estim_RANSAC=0;
    for(int i=0; i<img1Pts.size(); i++){        
        // computing the distance from point x' to epipolar line Fx
        Point2D p1=img1Pts[i];
        Point2D p2=img2Pts[i];
        double epidistance_estim=epidistance(p1 , p2, F);
        double epidistance_estim_RANSAC=epidistance(p1,p2,F_RANSAC.t());

        avg_epidist_estim+=epidistance_estim;
        avg_epidist_estim_RANSAC+=epidistance_estim_RANSAC;

        // std::cout << "epi distance estim: " << epidistance_estim << std::endl;
        // std:: cout << std::endl;
    }

    // std::cout << "avg epi distance estim: " << avg_epidist_estim/img1Pts.size() << std::endl;
    // std::cout << "avg epi distance estim RANSAC: " << avg_epidist_estim_RANSAC/img1Pts.size() << std::endl;

    // 4. After that, we want to do relative pose estimation given F and the intrinsic matrices K1 and K2:
    Mat P2=EstimatePose(I1,I2, K1, K2, F, img1Pts, img2Pts, R_gt, t_gt);
    
    // debugging
    // std::cout << "P2:" << std::endl;
    // printM(P2);

    // 5. From P2, we will extract R and t 
    Mat R_pred=P2.copy(0,2,0,2);
    Mat t_pred_mat=P2.copyCols(3,3);

    // std::cout << "t_pred-mat:" << std::endl;
    // printM(t_pred_mat);

    Vec t_pred(3);
    t_pred(0)=t_pred_mat(0,0);
    t_pred(1)=t_pred_mat(1,0);
    t_pred(2)=t_pred_mat(2,0);

    // debugging 
    // std::cout << "R_pred and t_pred:" << std::endl;
    // printM(R_pred);
    // printV(t_pred);

    // debugging: cheirality and reprojection test to test our R and t without gt
    int count=0;
    for(int i=0; i<img1Pts.size(); i++){        
        Vec u(2);
        Vec u_p(2);

        u(0)=img1Pts[i].x;
        u(1)=img1Pts[i].y;

        u_p(0)=img2Pts[i].x;
        u_p(1)=img2Pts[i].y;

        Mat Eye=Mat::zeros(3,4);
        Eye(0,0)=1.0;
        Eye(1,1)=1.0;
        Eye(2,2)=1.0;

        Mat P1_pixel=K1*Eye;
        Mat P2_pixel=K2*P2;

        if(Triangulate(I1, I2, u, u_p, P1_pixel, P2_pixel, R_pred, t_pred)>0) count+=1;
    }
    // std::cout << "Number of inliers: " << img1Pts.size() << std::endl;
    // std::cout << "Cheirality test result: " << count << std::endl;

    // debugging: average reprojection error using the obtained R and t: 
    // double total_squared_error = 0.0;
    // int point_count = 0;
    // for(int i=0; i<img1Pts.size(); i++){
    //     Vec u(2);
    //     Vec u_p(2);

    //     u(0)=img1Pts[i].x;
    //     u(1)=img1Pts[i].y;

    //     u_p(0)=img2Pts[i].x;
    //     u_p(1)=img2Pts[i].y;

    //     Mat Eye=Mat::zeros(3,4);
    //     Eye(0,0)=1.0;
    //     Eye(1,1)=1.0;
    //     Eye(2,2)=1.0;

    //     Mat P1_pixel=K1*Eye;
    //     Mat P2_pixel=K2*P2;

    //     total_squared_error+=ReprojectionError(u, u_p, P1_pixel, P2_pixel);
    //     point_count+=2;
    // }
    // std::cout << "The average reprojection error over all matches is: " << std::sqrt(total_squared_error / point_count) << std::endl;

    // 6. Finally, we print the Rotation and Translation error between the predicted and the ground truth:
    double rot_err=rotation_error(R_gt, R_pred);
    double trans_err=translation_error(t_gt, t_pred);

    // std::cout << "The Rotation error is: " << rot_err << std::endl; 
    // std::cout << "The Translation error is: " << trans_err << std::endl; 

    double coverage=computeGridCoverage(matches, I1.width(), I1.height());

    Vec info(4);
    info(0)=rot_err;
    info(1)=trans_err;
    info(2)=matches.size();
    info(3)=coverage;

    return info;
}
