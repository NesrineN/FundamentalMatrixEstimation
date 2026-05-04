#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <cmath>
#include "FNS.h"
#include "HEIV.h"
#include "Renorm.h"
#include "Initialization.h"
#include <vector>
#include <random>

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

double MatrixNorm(const Mat& A) {
    double sumSq = 0.0;
    // Loop through all rows and columns
    for (int i = 0; i < A.nrow(); i++) {
        for (int j = 0; j < A.ncol(); j++) {
            double val = A(i, j);
            sumSq += val * val;
        }
    }
    return std::sqrt(sumSq);
}

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

//////////////////////////////////////////////////////////////////////////////////////////////////////


// Main

int main()
{
    // Image setup

    int width  = 600;
    int height = 600;

    double f0 = 1200.0;

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
    Vec tplane1(0.0, 0.0, 4.0);

    auto plane1 = generatePlaneGrid(6, 6, 0.2, Rplane1, tplane1);
    
    double angle = 60.0 * PI / 180.0; // converting 60 degrees to radians 
    Mat Rplane2 = rotationY(angle);
    Vec tplane2(1.0, 0.0, 4.0);

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

    // First, trying without adding noise 
    // in order to test if the Fundamental Matrix Estimators work. 

    // Generating image correspondences

    std::vector<Point2D> img1Pts;
    std::vector<Point2D> img2Pts;

    double sigma = 1.0;

    std::mt19937 rng(0);

    for(const auto& X : worldPts)
    {
        Point2D p1 = projectPoint(X, K, Rc1, t1);

        Point2D p2 = projectPoint(X, K, Rc2, t2);

        // p1 = addNoise(p1, sigma, rng);
        // p2 = addNoise(p2, sigma, rng);

        img1Pts.push_back(p1);
        img2Pts.push_back(p2);
    }

    // // printing the correspondences:
    // for(int i=0; i<img1Pts.size(); i++){
    //     std::cout << "u1: " << img1Pts[i].x << " " << img1Pts[i].y << std::endl;
    //     std::cout << "u2: " << img2Pts[i].x << " " << img2Pts[i].y << std::endl;
    // }

    // Here, we normalize the image points: Hartley Normalization so that the centroid is at the origin and the average distance from origin is sqrt 2
    Mat T1=HartleyNormalize(img1Pts);
    Mat T2=HartleyNormalize(img2Pts);

    // Ground truth F
    Mat F_gt = computeGroundTruthF(K,K,Rc2,t2);
    
    // // checking if F_gt is correct
    // for(int i=0; i<img1Pts.size(); i++){
    //     Mat v1=Mat::zeros(3,1);
    //     Mat v2=Mat::zeros(3,1);

    //     v1(0)=img1Pts[i].x;
    //     v1(1)=img1Pts[i].y;
    //     v1(2)=1.0;

    //     v2(0)=img2Pts[i].x;
    //     v2(1)=img2Pts[i].y;
    //     v2(2)=1.0;

    //     // Should be very close to 0
    //     Mat v2tF=v2.t()*F_gt; // 1x3 matrix
    //     Mat error = v2tF*v1;
    //     std::cout << "Epipolar error: " << error(0) << std::endl;
    // }

    // it is correct

    // Estimating F

    // we create the Matrix Eall and vector of V0 matrices Vall
    // Eall is a 9xn matrix where each col corresponds to a point correspondence and so we have n columns in total --> n total point correspondences
    
    Mat Eall=Mat::zeros(9,img1Pts.size());
    std::vector<Mat> Vall;

    for(int i=0; i<img1Pts.size(); i++){
        Point2D p1=img1Pts[i];
        Point2D p2=img2Pts[i];

        Vec x1(3);
        x1(0)=p1.x;
        x1(1)=p1.y;
        x1(2)=1;

        Vec x2(3);
        x2(0)=p2.x;
        x2(1)=p2.y;
        x2(2)=1;

        Vec x1_norm=T1*x1;
        Vec x2_norm=T2*x2;

        Point2D p1_norm;
        Point2D p2_norm;

        p1_norm.x=x1_norm(0)/x1_norm(2);
        p1_norm.y=x1_norm(1)/x1_norm(2);

        p2_norm.x=x2_norm(0)/x2_norm(2);
        p2_norm.y=x2_norm(1)/x2_norm(2);

        Vec E=fillE(p1_norm,p2_norm,1);
        // Vec E=fillE(p1,p2,f0);

        for(int j = 0; j < 9; ++j)
        {
            Eall(j, i) = E(j);
        }

        // V0:
        Mat V0=Mat::zeros(9);

        double x=p1_norm.x;
        double y=p1_norm.y;
        double xp=p2_norm.x;
        double yp=p2_norm.y;

        // double x=p1.x;
        // double y=p1.y;
        // double xp=p2.x;
        // double yp=p2.y;

        f0=1.0;

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

        // adding V0 to the list Vall
        Vall.push_back(V0);
    }

    // we initialize uinit using Taubin method
    Vec uinit= Taubin(Eall);
    // Vec uinit=RandomInit();
    // Vec uinit = LeastSquares(Eall);

    // we estimate F using FNS method
    // Mat F =FNS(uinit, Eall, Vall);

    Vec vinit=uinit.copy(0,7);

    Mat F =HEIV(vinit, Eall, 1.0);

    // Finally, we de-normalize F:
    F=(T2.t()*F)*T1;

    // // checking if F estimated is correct
    for(int i=0; i<img1Pts.size(); i++){
        Mat v1=Mat::zeros(3,1);
        Mat v2=Mat::zeros(3,1);

        v1(0)=img1Pts[i].x;
        v1(1)=img1Pts[i].y;
        v1(2)=1.0;

        v2(0)=img2Pts[i].x;
        v2(1)=img2Pts[i].y;
        v2(2)=1.0;

        // Should be very close to 0
        Mat v2tF=v2.t()*F; // 1x3 matrix
        Mat error = v2tF*v1;
        std::cout << "Epipolar error: " << error(0) << std::endl;
    }

    // Printing the results

    std::cout << "Ground Truth Fundamental Matrix:\n";

    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            std::cout << F_gt(i,j) << " ";
        }

        std::cout << std::endl;
    }

    std::cout << "Estimated Fundamental Matrix:\n";

    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            std::cout << F(i,j) << " ";
        }

        std::cout << std::endl;
    }

    return 0;
}