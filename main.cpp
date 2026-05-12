#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <cmath>
#include "FNS.h"
#include "HEIV.h"
#include "Renorm.h"
#include "GaussNewton.h"
#include "Initialization.h"
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

const double PI = 3.14159265358979323846;

// we will create 2 simulated images of two planar grid planes joined at angle 60◦. Each image is of size 600x600 with f0=1200 pixels.

// we need to create the 3d scene of the 2 planar grid planes and then project the scene onto 2 images: left and right. 
// for the projection, we need to define the cameras' intrinsics 

// after projection , we add gaussian noise to each of the 600 pixels of each image


//////////////////////////////////////////////////////////////////////////////////////////////////////

struct Point2D
{
    double x;
    double y;
};

// Rotation around Y axis

Mat rotationY(double angle)
{
    Mat R(3,3);

    double c = std::cos(angle);
    double s = std::sin(angle);

    R(0,0) =  c;  R(0,1) = 0;  R(0,2) = s;
    R(1,0) =  0;  R(1,1) = 1;  R(1,2) = 0;
    R(2,0) = -s;  R(2,1) = 0;  R(2,2) = c;

    return R;
}

// Projecting 3D point into image

Point2D projectPoint(const Vec& X, const Mat& K, const Mat& R, const Vec& t)
{
    Vec Xc = R * X + t;

    Vec x = K * Xc; // image coordinates homogeneous coordinates 

    Point2D p;

    p.x = x(0) / x(2);
    p.y = x(1) / x(2);

    return p;
}

// computing the ground truth of the fundamental matrix

// returns the skew symmetric matrix of t
Mat skew(const Vec& t){
    Mat S(3,3);

    S(0,0) = 0;
    S(0,1) = -t(2);
    S(0,2) =  t(1);

    S(1,0) =  t(2);
    S(1,1) = 0;
    S(1,2) = -t(0);

    S(2,0) = -t(1);
    S(2,1) =  t(0);
    S(2,2) = 0;

    return S;
}

Mat computeGroundTruthF(const Mat& Kl, const Mat& Kr, const Mat& R, const Vec& t){
    Mat Klinv=Kl.inv();
    Mat Krinv=Kr.inv();
    Mat Tx= skew(t);

    Mat F = Klinv.t() * Tx * R * Krinv;

    return F;
}

// function that creates the vector E from two point-correspondences:
Vec fillE(const Point2D& p1, const Point2D& p2, double f0, double cx, double cy){
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

    // double x  = p1.x;
    // double y  = p1.y;
    // double xp = p2.x;
    // double yp = p2.y;

    // double s = f0;

    // Vec E(9);

    // E(0) = (x * xp) / (s * s);
    // E(1) = (x * yp) / (s * s);
    // E(2) = (x) / s;

    // E(3) = (y * xp) / (s * s);
    // E(4) = (y * yp) / (s * s);
    // E(5) = (y) / s;

    // E(6) = (xp) / s;
    // E(7) = (yp) / s;
    // E(8) = 1.0;

    // return E;
}



// Hartley Normalization of the image points:
Mat HartleyNormalize(const std::vector<Point2D>& imgpoints){
    double xbar=0;
    double ybar=0;

    for(const auto& X : imgpoints){
        double x=X.x;
        double y=X.y;
        xbar+=x;
        ybar+=y;
    }

    xbar/=imgpoints.size();
    ybar/=imgpoints.size();

    std::vector<double> xtilde;
    std::vector<double> ytilde;

    for(const auto& X : imgpoints){
        double x=X.x;
        double y=X.y;
        
        xtilde.push_back(x-xbar);
        ytilde.push_back(y-ybar);
    }

    double distance=0;

    for(int i=0; i<xtilde.size(); i++){
        distance+= std::sqrt((xtilde[i]*xtilde[i])+(ytilde[i]*ytilde[i]));
    }

    distance/=xtilde.size();

    if(distance<1e-10){distance+=1e-6;}

    double s=std::sqrt(2)/distance;

    Mat T=Mat::zeros(3);
    T(0,0)=s;
    T(1,1)=s;
    T(2,2)=1;

    T(0,2)=-s*xbar;
    T(1,2)=-s*ybar;

    return T;
    
}

// Adding Gaussian noise

Point2D addNoise(const Point2D& p, double sigma, std::mt19937& rng)
{
    std::normal_distribution<double> dist(0.0, sigma);

    Point2D noisyp;

    noisyp.x = p.x + dist(rng);
    noisyp.y = p.y + dist(rng);

    return noisyp;
}

// Generating the planar grid

std::vector<Vec> generatePlaneGrid(int rows, int cols, double spacing, const Mat& R, const Vec& t)
{
    std::vector<Vec> pts;

    for(int i = 0; i < rows; ++i)
    {
        for(int j = 0; j < cols; ++j)
        {
            Vec p(3);

            p(0) = j * spacing;
            p(1) = i * spacing;
            p(2) = 0.0;

            p = R * p + t;

            pts.push_back(p);
        }
    }

    return pts;
}

Mat computeV0(const Mat& Eall){
    Mat V0 = Mat::zeros(9, 9);

    for (int k=0; k<Eall.ncol(); k++)
    {   Vec E=Eall.col(k);
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                V0(i, j) += E(i) * E(j);
            }
        }
    }

    return V0;

}

// Mat computeV0(double x, double y, double xp, double yp, double f0){

//     Mat V0=Mat::zeros(9);
//     double s = f0;
    
//     // R0
//     V0(0,0)= (x*x) + (xp*xp);
//     V0(0,1)= xp*yp;
//     V0(0,2)= f0 * xp;
//     V0(0,3)= x*y;
//     V0(0,4)= 0;
//     V0(0,5)=0;
//     V0(0,6)=f0*x;
//     V0(0,7)=0;
//     V0(0,8)=0;

//     // R1
//     V0(1,0)= xp*yp;
//     V0(1,1)= x*x + yp*yp;
//     V0(1,2)= f0 * yp;
//     V0(1,3)= 0;
//     V0(1,4)= x*y;
//     V0(1,5)=0;
//     V0(1,6)=0;
//     V0(1,7)=f0*x;
//     V0(1,8)=0;

//     // R2
//     V0(2,0)= f0*xp;
//     V0(2,1)= f0*yp;
//     V0(2,2)= f0 * f0;
//     V0(2,3)= 0;
//     V0(2,4)= 0;
//     V0(2,5)=0;
//     V0(2,6)=0;
//     V0(2,7)=0;
//     V0(2,8)=0;

//     // R3
//     V0(3,0)= x*y;
//     V0(3,1)= 0;
//     V0(3,2)= 0;
//     V0(3,3)= y*y + xp*xp;
//     V0(3,4)= xp*yp;
//     V0(3,5)= f0*xp;
//     V0(3,6)= f0*y;
//     V0(3,7)=0;
//     V0(3,8)=0;

//     // R4
//     V0(4,0)= 0;
//     V0(4,1)= x*y;
//     V0(4,2)= 0;
//     V0(4,3)= xp*yp;
//     V0(4,4)= y*y + yp*yp;
//     V0(4,5)= f0*yp;
//     V0(4,6)= 0;
//     V0(4,7)=f0*y;
//     V0(4,8)=0;

//     // R5
//     V0(5,0)= 0;
//     V0(5,1)= 0;
//     V0(5,2)= 0;
//     V0(5,3)= f0*xp;
//     V0(5,4)= f0*yp;
//     V0(5,5)= f0*f0;
//     V0(5,6)= 0;
//     V0(5,7)=0;
//     V0(5,8)=0;

//     // R6
//     V0(6,0)= f0*x;
//     V0(6,1)= 0;
//     V0(6,2)= 0;
//     V0(6,3)= f0*y;
//     V0(6,4)= 0;
//     V0(6,5)= 0;
//     V0(6,6)= f0*f0;
//     V0(6,7)=0;
//     V0(6,8)=0;

//     // R7
//     V0(7,0)= 0;
//     V0(7,1)= f0*x;
//     V0(7,2)= 0;
//     V0(7,3)= 0;
//     V0(7,4)= f0*y;
//     V0(7,5)= 0;
//     V0(7,6)= 0;
//     V0(7,7)= f0*f0;
//     V0(7,8)=0;

//     // R8
//     V0(8,0)= 0;
//     V0(8,1)= 0;
//     V0(8,2)= 0;
//     V0(8,3)= 0;
//     V0(8,4)= 0;
//     V0(8,5)= 0;
//     V0(8,6)= 0;
//     V0(8,7)= 0;
//     V0(8,8)=0;

//     return V0;

// }

double F_error(const Vec& u, const Vec& u_hat){
    // normalization at first just in case:

    Vec u1=u;
    u1/=std::sqrt(u1.qnorm());

    Vec u2=u_hat;
    u2/=std::sqrt(u2.qnorm());

    Mat I=Mat::eye(9);
    Mat P=I-(u1*u1.t()); 
    Vec Puhat=P*u2;

    return std::sqrt(Puhat.qnorm());
}

double sampsonError(const Mat& v1, const Mat& v2, const Mat& F)
{
    // f = x2^T F x1
    double f =
        v2(0)*(F(0,0)*v1(0) + F(0,1)*v1(1) + F(0,2)*v1(2)) +
        v2(1)*(F(1,0)*v1(0) + F(1,1)*v1(1) + F(1,2)*v1(2)) +
        v2(2)*(F(2,0)*v1(0) + F(2,1)*v1(1) + F(2,2)*v1(2));

    // Fx1
    double Fx1_0 = F(0,0)*v1(0) + F(0,1)*v1(1) + F(0,2)*v1(2);
    double Fx1_1 = F(1,0)*v1(0) + F(1,1)*v1(1) + F(1,2)*v1(2);
    double Fx1_2 = F(2,0)*v1(0) + F(2,1)*v1(1) + F(2,2)*v1(2);

    // F^T x2
    double Ftx2_0 = F(0,0)*v2(0) + F(1,0)*v2(1) + F(2,0)*v2(2);
    double Ftx2_1 = F(0,1)*v2(0) + F(1,1)*v2(1) + F(2,1)*v2(2);
    double Ftx2_2 = F(0,2)*v2(0) + F(1,2)*v2(1) + F(2,2)*v2(2);

    double denom =
        Fx1_0*Fx1_0 + Fx1_1*Fx1_1 + Fx1_2*Fx1_2 +
        Ftx2_0*Ftx2_0 + Ftx2_1*Ftx2_1 + Ftx2_2*Ftx2_2;

    return (f * f) / denom;
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

// returns the root-mean-squares of ||Puuˆ|| over 1000 trials for the method chosen. 1==FNS, 2==HEIV, 3==Renorm, 4==GaussNewton  
double F_Test(int method, double sigma, const Vec& u_gt, const std::vector<Vec>& worldPts, const Mat& K, const Mat& Rc1, const Vec& t1, const Mat& Rc2, const Vec& t2, double f0, double cx, double cy, std::mt19937& rng){
    // Generating noisy image correspondences
    double error=0;
    int total_trials=1;

    for(int trial=0; trial<total_trials; trial++){
        std::vector<Point2D> img1Pts;
        std::vector<Point2D> img2Pts;

        for(const auto& X : worldPts)
        {
            Point2D p1 = projectPoint(X, K, Rc1, t1);

            Point2D p2 = projectPoint(X, K, Rc2, t2);   

            // p1 = addNoise(p1, sigma, rng);
            // p2 = addNoise(p2, sigma, rng);

            img1Pts.push_back(p1);
            img2Pts.push_back(p2);
        }

        // printing the correspondences:
        // for(int i=0; i<img1Pts.size(); i++){
        //     std::cout << "u1: " << img1Pts[i].x << " " << img1Pts[i].y << std::endl;
        //     std::cout << "u2: " << img2Pts[i].x << " " << img2Pts[i].y << std::endl;
        // }

        // we create the Matrix Eall and vector of V0 matrices Vall
        // Eall is a 9xn matrix where each col corresponds to a point correspondence and so we have n columns in total --> n total point correspondences
        
        Mat Eall=Mat::zeros(9,img1Pts.size());
        std::vector<Mat> Vall;

        for(int i=0; i<img1Pts.size(); i++){
            Point2D p1=img1Pts[i];
            Point2D p2=img2Pts[i];

            Vec E=fillE(p1,p2,1.0, cx, cy);

            for(int j = 0; j < 9; ++j)
            {
                Eall(j, i) = E(j);
            }

            double x=p1.x;
            double y=p1.y;
            double xp=p2.x;
            double yp=p2.y;

            // Mat V0=computeV0(x,y,xp,yp,1.0);
            Mat V0=computeV0(Eall);

            // adding V0 to the list Vall
            Vall.push_back(V0);
        }

        // we initialize uinit using Taubin method
        Vec uinit= Taubin(Eall, f0, Vall);
        Vec vinit=uinit.copy(0,7);

        // Mat F=Mat::zeros(3);

        // switch (method)
        // {
        // case 1:
        //     // FNS
            // F =FNS(uinit, Eall, Vall);
        //     break;

        // case 2:
        //     // HEIV
        //     F =HEIV(vinit, Eall, f0);
        //     break;

        // case 3:
        //     // Renorm
        //     F = Renorm(uinit, Eall, Vall); 
        //     break;
        
        // case 4:
        //     // GaussNewton
        //     F = GaussNewton(uinit, Eall, Vall);
        //     break;
        // }

        Mat F=Mat::zeros(3);
        for(int i=0; i<3; i++){
            for(int j=0; j<3; j++){
                F(i, j) = uinit(i * 3 + j); // Swapped i and j
            }
        }

        printM(F);

        // checking if F estimated is correct
        std::vector<double> errors;
        for(int i=0; i<img1Pts.size(); i++){
            Mat v1=Mat::zeros(3,1);
            Mat v2=Mat::zeros(3,1);

            v1(0)=img1Pts[i].x;
            v1(1)=img1Pts[i].y;
            v1(2)=f0;

            v2(0)=img2Pts[i].x;
            v2(1)=img2Pts[i].y;
            v2(2)=f0;

            // Should be very close to 0
            Mat v2tF=v2.t()*F; // 1x3 matrix
            Mat error = v2tF*v1;
            std::cout << "Epipolar error: " << error(0) << std::endl;

            double sam=sampsonError(v1, v2, F);
            errors.push_back(sam);
            // std::cout << "Epipolar error: " << sam << std::endl;

        }

        std::sort(errors.begin(), errors.end());
        for(int k=0; k<errors.size(); k++){
            // std::cout << "sam: " << errors[k] << std::endl;
        }

        Vec u_estimated(9);
        u_estimated(0)=F(0,0);
        u_estimated(1)=F(0,1);
        u_estimated(2)=F(0,2);
        u_estimated(3)=F(1,0);
        u_estimated(4)=F(1,1);
        u_estimated(5)=F(1,2);
        u_estimated(6)=F(2,0);
        u_estimated(7)=F(2,1);
        u_estimated(8)=F(2,2);

        u_estimated/=std::sqrt(u_estimated.qnorm());

        // we add the error obtained to the sum of the errors:
        double e=F_error(u_gt, u_estimated);
        error += e * e;

        // std::cout << "Trial " << trial << " finished!" << std::endl;
    }

    return std::sqrt(error/double(total_trials));

}

//////////////////////////////////////////////////////////////////////////////////////////////////////

// Main

int main()
{
    // Image setup

    int width  = 600;
    int height = 600;

    double f0 = 1200.0;

    std::mt19937 rng(0);

    Mat K(3,3);

    K(0,0) = f0;
    K(0,1) = 0;
    K(0,2) = width / 2.0;

    K(1,0) = 0;
    K(1,1) = f0;
    K(1,2) = height / 2.0;

    K(2,0) = 0;
    K(2,1) = 0;
    K(2,2) = 1;

    // Creating two planes joined at 60 degrees

    Mat Rplane1 = Mat::eye(3);
    Vec tplane1(0.0, 0.0, 8.0);

    auto plane1 = generatePlaneGrid(6, 6, 0.2, Rplane1, tplane1);
    
    double angle = 60.0 * PI / 180.0; // converting 60 degrees to radians 
    Mat Rplane2 = rotationY(angle);
    Vec tplane2(1.0, 0.0, 8.0);

    auto plane2 = generatePlaneGrid(6,6,0.2,Rplane2,tplane2);

    std::vector<Vec> worldPts;

    worldPts.insert(worldPts.end(),plane1.begin(),plane1.end());

    worldPts.insert(worldPts.end(),plane2.begin(),plane2.end());

    // Camera 1 , also world 

    Mat Rc1 = Mat::eye(3); // world axes
    Vec c1(0.0, 0.0, 0.0); // world origin
    Vec t1= - Rc1*c1; // 0

    // Camera 2

    Mat Rc2 =rotationY(5.0 * PI / 180.0); // if we apply Rc2 we would transform the world/cam1 axes to c2's axes
    Vec c2(0.5, 0.0, 0.0); // where c2 is with respect to the world origin also cam 1

    Vec t2 = -Rc2 * c2;

    std::vector<Point2D> img1Pts_gt;
    std::vector<Point2D> img2Pts_gt;

    for(const auto& X : worldPts)
    {
        Point2D p1 = projectPoint(X, K, Rc1, t1);

        Point2D p2 = projectPoint(X, K, Rc2, t2);   
        
        img1Pts_gt.push_back(p1);
        img2Pts_gt.push_back(p2);
    }

    // printing the correspondences:
    // for(int i=0; i<img1Pts_gt.size(); i++){
    //     std::cout << "u1: " << img1Pts_gt[i].x << " " << img1Pts_gt[i].y << std::endl;
    //     std::cout << "u2: " << img2Pts_gt[i].x << " " << img2Pts_gt[i].y << std::endl;
    // }

    // Ground truth F
    Mat F_gt = computeGroundTruthF(K,K,Rc2,t2);

    // normalizing F_gt: 
    Vec u_gt(9);
    u_gt(0)=F_gt(0,0);
    u_gt(1)=F_gt(0,1);
    u_gt(2)=F_gt(0,2);
    u_gt(3)=F_gt(1,0);
    u_gt(4)=F_gt(1,1);
    u_gt(5)=F_gt(1,2);
    u_gt(6)=F_gt(2,0);
    u_gt(7)=F_gt(2,1);
    u_gt(8)=F_gt(2,2);

    u_gt/=std::sqrt(u_gt.qnorm());

    // checking if F gt is correct
    // for(int i=0; i<img1Pts_gt.size(); i++){
    //     Mat v1=Mat::zeros(3,1);
    //     Mat v2=Mat::zeros(3,1);

    //     v1(0)=img1Pts_gt[i].x;
    //     v1(1)=img1Pts_gt[i].y;
    //     v1(2)=1.0;

    //     v2(0)=img2Pts_gt[i].x;
    //     v2(1)=img2Pts_gt[i].y;
    //     v2(2)=1.0;

    //     // Should be very close to 0
    //     Mat v2tF=v2.t()*F_gt; // 1x3 matrix
    //     Mat error = v2tF*v1;
    //     std::cout << "Epipolar error: " << error(0) << std::endl;
    // }

    // FNS trial:
    double cx=width/2.0;
    double cy=height/2.0;

    double error=F_Test(1, 1, u_gt, worldPts,K, Rc1,t1,  Rc2, t2, f0, cx, cy,  rng);

    std::cout << "Error is: " << error << std::endl;

    // std::vector<double> FNS_errors;
    // std::vector<double> HEIV_errors;
    // std::vector<double> Renorm_errors;
    // std::vector<double> GaussNewton_errors;

    // σ ∈ {0.001, 0.0025, 0.005, 0.01, 0.02, 0.05}
    // If you want to reproduce their plots:

    // use pixel coordinates (not normalized)
    // σ ∈ [0, 9]
    // 600×600 image
    // f0 = 1200
    // no Hartley normalization

    // If you want modern ML-style evaluation:

    // use Hartley normalization
    // σ ∈ [0.001, 0.05]

    // for(int sigma=1; sigma<10; sigma++){
        // FNS:
        // double FNS_error=F_Test(1, sigma, u_gt, worldPts, K, Rc1, t1, Rc2, t2, f0, rng);
        // FNS_errors.push_back(FNS_error);
        // std::cout << "Sigma " << sigma << std::endl;

        // HEIV:
        // double HEIV_error=F_Test(2, sigma, u_gt, worldPts, K, Rc1, t1, Rc2, t2, f0, rng);
        // HEIV_errors.push_back(HEIV_error);
        // std::cout << "Sigma " << sigma << std::endl;

        // Renorm:
        // double Renorm_error=F_Test(3, sigma, u_gt, worldPts, K, Rc1, t1, Rc2, t2, f0, rng);
        // Renorm_errors.push_back(Renorm_error);
        // std::cout << "Sigma " << sigma << std::endl;

        // Gauss Newton:
        // double GaussNewton_error=F_Test(4, sigma, F_gt, worldPts, K, Rc1, t1, Rc2, t2, f0, rng);
        // GaussNewton_errors.push_back(GaussNewton_error);
    // }

    // std::ofstream file("FNS_errors.csv");

    // if (!file.is_open()) {
    //     std::cerr << "Failed to open file\n";
    //     return 1;
    // }

    // // Optional header
    // file << "FNS error\n";

    // for (double v : FNS_errors) {
    //     file << v << "\n";
    // }

    // file.close();
    
    return 0;
}