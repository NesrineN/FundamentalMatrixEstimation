#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <cmath>
#include "FNS.h"
#include "HEIV.h"
#include "RANSAC.h"
#include "Renorm.h"
#include "GaussNewton.h"
#include "Initialization.h"
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include "PoseEstimation.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

const double PI = 3.14159265358979323846;

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

using namespace Imagine;
using namespace std;

struct Point2D
{
    double x;
    double y;
};

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

Mat GetF(const std::vector<Point2D>& img1Pts, const std::vector<Point2D>& img2Pts, const double& f0, const int& method){
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
    Vec uinit= Taubin(Eall, f0, Vall);

    Mat F=Mat::zeros(3);

    if(method==1){
        // we run FNS to get F
        F =FNS(uinit, Eall, Vall);
    }
    else{
        F = GaussNewton(uinit, Eall, Vall);
    }
    
    F=F.t();

    return F;
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
    for(int i=0;i<V.ncol(); i++){
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

    double c = std::abs(dot(a, b));

    c = clamp(c, -1.0, 1.0);

    double angle = std::acos(c);

    return angle * 180.0 / PI;
}

Vec RunPipelineNoiseless(const std::string& I1_path, const std::string& I2_path, const Mat& K1, const Mat& K2, const double& f0, const int& method=1, const Mat& R_gt, const Vec& t_gt){

    // 1. Get the matches/inliers from the image pairs using SIFT + RANSAC and 8-point algorithm as a starting point
    vector<Match> matches=GetInliers(I1_path, I2_path);

    // 2. Using the matches, we want to fill the vectors of 2D point matches img1Pts and img2Pts
    std::vector<Point2D> img1Pts;
    std::vector<Point2D> img2Pts;
    for(int i=0; i<matches.size(); i++){
        Point2D p1;
        Point2D p2;

        p1.x=matches[i].x1;
        p1.y=matches[i].y1;

        p2.x=matches[i].x2
        p2.y=matches[i].y2;

        img1Pts.push_back(p1);
        img2Pts.push_back(p2);
    }

    // 3. Now, we want to run the method of choice: either FNS or Gauss-Newton to compute the fundamental matrix from the inliers
    Mat F=GetF(img1Pts, img2Pts, f0, method);

    // 4. After that, we want to do relative pose estimation given F and the intrinsic matrices K1 and K2:
    Mat P2=EstimatePose(K1, K2, F, img1Pts, img2Pts);
    
    // debugging
    printM(P2);

    // 5. From P2, we will extract R and t 
    Mat R_pred=P2.copy(0,2,0,2);
    Vec t_pred=P2.copyCols(3,3);

    // debugging 
    printM(R_pred);
    printV(t_pred);

    // 6. Finally, we print the Rotation and Translation error between the predicted and the ground truth:
    double rot_err=rotation_error(R_gt, R_pred);
    double trans_err=translation_error(t_gt, t_pred);

    std::cout << "The Rotation error is: " << rot_err << std::endl; 
    std::cout << "The Translation error is: " << trans_err << std::endl; 

    Vec errors(2);
    errors(0)=rot_err;
    errors(1)=trans_err;

    return errors;
}
