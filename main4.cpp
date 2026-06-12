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

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

using namespace Imagine;
using namespace std;

// ----------------------------------------------------------------------

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


// ---------------------------------------------------------------------

// might need to normalize the inliers?


int main(int argc, char* argv[])
{
    srand((unsigned int)time(0));

    std::string s1 = argc>1? argv[1]: srcPath("im1111.png");
    std::string s2 = argc>2? argv[2]: srcPath("im2222.png");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1.c_str()) ||
        ! load(I2, s2.c_str()) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }

    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    // -----------------------------------------------------------------------
    double f0 = 1200.0;

    std::vector<Point2D> img1Pts;
    std::vector<Point2D> img2Pts;
    // vector<Match> matches;

    // getting the inliers from the inliers.txt file provided in the demo of the IPOL journal article: Fundamental Matrix of a Stereo Pair, with A Contrario Elimination of Outliers
    
    std::ifstream file("inliers4.txt");

    if (!file.is_open()) {
        std::cerr << "Failed to open file\n";
        return 1;
    }

    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        double x,y,xp,yp;

        while (ss >> x >> y >> xp >> yp) {            
            Point2D p1;
            Point2D p2;
            // Match m;

            p1.x=x;
            p1.y=y;

            p2.x=xp;
            p2.y=yp;

            // m.x1=x;
            // m.y1=y;
            // m.x2=xp;
            // m.y2=yp;

            img1Pts.push_back(p1);
            img2Pts.push_back(p2);
            // matches.push_back(m);
        }
    }

    // printing the correspondences:
    // std::cout << "correspondences: " << std::endl;
    // for(int i=0; i<img1Pts.size(); i++){
    //     std::cout << "u1: " << img1Pts[i].x << " " << img1Pts[i].y << std::endl;
    //     std::cout << "u2: " << img2Pts[i].x << " " << img2Pts[i].y << std::endl;
    // }

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

    Vec vinit=uinit.copy(0,7);


    // Mat F=Mat::zeros(3);
    // for(int i=0; i<3; i++){
    //     for(int j=0; j<3; j++){
    //         F(i, j) = uinit(i * 3 + j); 
    //     }
    // }

    Mat F=Mat::zeros(3);

    // F =HEIV(vinit, Eall, f0);
    F =FNS(uinit, Eall, Vall);
    // F = Renorm(uinit, Eall, Vall); 
    // F = GaussNewton(uinit, Eall, Vall);

    // observation:avg distance to epipolar line error for all methods is relatively the same except for HEIV which is much higher. the same was observed for 2 different examples of image pairs. 
    // avg error for all 3 methods for images im11 and im22: around 130. for HEIV around 743
    // avg error for all 3 methods for images im1 and im2: around 83. for HEIV around 841
    // avg error for all 3 method for images im111 and im222: around 18 - 19 . for HEIV around 110.
    // avg error for all 3 method for images im1111 and im2222: around 113-141-143. for HEIV around 370.

    // all methods should yield similar results. the comparison is convergence time. must fix HEIV!!
    // problem: sometimes Taubin yielded better results than the other methods ! 

    // de-normalizing F to be closer to F_gt:
    Mat Norm=Mat::eye(3);
    Norm(2,2)=f0;

    Mat F_denorm=Norm.t()*F*Norm;

    F=F.t();
    F_denorm=F_denorm.t();

    double avg_epidist_estim=0;

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

        
        // computing the distance from point x' to epipolar line Fx
        Point2D p1=img1Pts[i];
        Point2D p2=img2Pts[i];
        double epidistance_estim=epidistance(p1 , p2, F_denorm);

        avg_epidist_estim+=epidistance_estim;

        // std::cout << "epi distance estim: " << epidistance_estim << std::endl;
        // std:: cout << std::endl;
    }

    std::cout << "avg epi distance estim: " << avg_epidist_estim/img1Pts.size() << std::endl;

    // Redisplay without matches
    display(I1,0,0);
    display(I2,w,0);
    // click at any point and in an image and will display its corresponding epipolar line in the other image

    FMatrix<float,3,3> F_denorm_2;

    for(int i=0;i<3;i++){
        for(int j=0; j<3; j++){
            F_denorm_2(i,j)=F_denorm(i,j);
        }
    }

    displayEpipolar(I1, I2, F_denorm_2);

    endGraphics();
    return 0;
}
