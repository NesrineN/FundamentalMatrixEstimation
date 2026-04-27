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

// returns the skew symmetric matric of t
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

Mat computeGroundTruthF(const Mat& K, const Mat& R, const Vec& t){
    Mat Kinv=K.inv();
    Mat Tx= skew(t);

    Mat F = Kinv.t() * Tx * R * Kinv;

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

    // Camera 1

    Mat Rc1 = Mat::eye(3);
    Vec tc1(0.0, 0.0, 0.0);

    // Camera 2

    Mat Rc2 =rotationY(5.0 * PI / 180.0);
    Vec tc2(-0.5, 0.0, 0.0);

    // First, trying with sigma = 0 : no noise
    // in order to test if the Fundamental Matrix Estimators work. 

    // Generating image correspondences

    std::vector<Point2D> img1Pts;
    std::vector<Point2D> img2Pts;

    double sigma = 0.0;

    std::mt19937 rng(0);

    for(const auto& X : worldPts)
    {
        Point2D p1 = projectPoint(X, K, Rc1, tc1);

        Point2D p2 = projectPoint(X, K, Rc2, tc2);

        p1 = addNoise(p1, sigma, rng);
        p2 = addNoise(p2, sigma, rng);

        img1Pts.push_back(p1);
        img2Pts.push_back(p2);
    }


    // Ground truth F
    Mat F_gt = computeGroundTruthF(K,Rc2,tc2); 

    // Estimating F

    // testing with a random initial guess of u:
    Vec uinit= RandomInit();

    // next we create the Matrix Eall
    // Eall is a 9xn matrix where each col corresponds to a point correspondence and so we have n columns in total --> n total point correspondences
    
    Mat Eall=Mat::zeros(9,img1Pts.size());

    for(int i=0; i<img1Pts.size(); i++){
        Point2D p1=img1Pts[i];
        Point2D p2=img2Pts[i];

        E=fillE(p1,p2,f0);

        for(int j = 0; j < 9; ++j)
        {
            Eall(j, i) = E(j);
        }
        
    }
    
    Mat F =FNS(uinit, Eall);

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