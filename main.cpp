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

struct Match {
    double x1, y1, x2, y2;
};

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

vector<FMatrix<float,3,3>> compute_N(vector<Match>& matches){
    
    vector<FMatrix<float,3,3>> N_list;
    FMatrix<float,3,3> N1; // the Normalization matrix for Image 1
    FMatrix<float,3,3> N2; // the Normalization matrix for Image 2

    // we will calculate the centroid of points of each image, then the distance from all points of an image to its centroid, get the scale and fill the matrix with it and the centroid
    // the aim is to translate the points so their centroid is at the origin and to scale the points so that the average distance from the origin is sqrt(2). 

    // for I1:
    float xbar1=0.f;
    float ybar1=0.f;
    for(int j=0;j<matches.size();j++){
        xbar1=xbar1+matches[j].x1;
        ybar1=ybar1+matches[j].y1;
    }
    xbar1/=(float)matches.size(); // centroid
    ybar1/=(float)matches.size(); // centroid

    float distance1=0.f;

    for(int ii=0;ii<matches.size();ii++){
        distance1=distance1+(sqrt(pow((matches[ii].x1-xbar1),2)+pow((matches[ii].y1-ybar1),2)));
    }

    distance1/=(float)matches.size(); // distance of points from the centroid
    if(distance1 < 1e-8f) distance1 = 1.0f; // to avoid dividing by zero in case distance is very close to zero
    float s1=sqrt(2.0f)/distance1; // scale to normalize by
    
    N1.fill(0.0f);
    N1(0,0)=s1;
    N1(0,1)=0;
    N1(0,2)=-s1*xbar1;
    N1(1,0)=0;
    N1(1,1)=s1;
    N1(1,2)=-s1*ybar1;
    N1(2,0)=0;
    N1(2,1)=0;
    N1(2,2)=1;

    N_list.push_back(N1);

    // for I2:
    float xbar2=0.f;
    float ybar2=0.f;
    for(int id=0;id<matches.size();id++){
        xbar2=xbar2+matches[id].x2;
        ybar2=ybar2+matches[id].y2;
    }
    xbar2/=(float)matches.size();
    ybar2/=(float)matches.size();
    float distance2=0.f;
    for(int ind=0;ind<matches.size();ind++){
        distance2=distance2+(sqrt(pow((matches[ind].x2-xbar2),2)+pow((matches[ind].y2-ybar2),2)));
    }
    distance2/=(float)matches.size();
    if(distance2 < 1e-8f) distance2 = 1.0f;    
    float s2=sqrt(2.0f)/distance2;

    N2.fill(0.0f);
    N2(0,0)=s2;
    N2(0,1)=0;
    N2(0,2)=-s2*xbar2;
    N2(1,0)=0;
    N2(1,1)=s2;
    N2(1,2)=-s2*ybar2;
    N2(2,0)=0;
    N2(2,1)=0;
    N2(2,2)=1;

    N_list.push_back(N2);


    return N_list;

}

// Function that takes the matches and the two normalization matrices N1 for I1 and N2 for I2 and normalizes the matches, returns the normalized matches
vector<Match> normalize_matches(FMatrix<float,3,3> N1, FMatrix<float,3,3> N2, vector<Match>& matches){
    
    vector<Match> subset_normalized;
    for(int match_id=0;match_id<matches.size(); match_id++){
        Match m=matches[match_id];
        Match m_normalized;

        FVector<float,3> X1h, X2h; // point correspondences in a match in homogeneous system
        X1h[0]= m.x1;  X1h[1]= m.y1;  X1h[2]= 1.0f;
        X2h[0]= m.x2;  X2h[1]= m.y2;  X2h[2]= 1.0f;

        // we normalize both
        FVector<float,3> X1n= N1 * X1h;
        FVector<float,3> X2n= N2 * X2h;

        // we assign to m_normalized after returning to euclidean 2d:
        m_normalized.x1= X1n[0] / X1n[2];
        m_normalized.y1= X1n[1] / X1n[2];

        m_normalized.x2= X2n[0] / X2n[2];
        m_normalized.y2= X2n[1] / X2n[2];

        subset_normalized.push_back(m_normalized);
    }

    return subset_normalized;

}

void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float,3,3>& F) {
    while(true) {
        int x,y; // coordinates of point clicked
        if(getMouse(x,y) == 3)
            break;
        // --------------- TODO ------------
        // if point clicked in I1 --> point x and matching point is x' which lies on the line F.x
        // if point clicked in I2 --> point x' and matching point is x which lies on the line F^Tx'

        if(x>=0 && x<I1.width() && y>=0 && y<I1.height()){
            cout << "You clicked a point in I1" << endl;
            // user clicked point in I1
            FVector<float,3> l; // this is the epipolar line F.x
            FVector<float,3> Xh; // this is the point clicked x in homogeneous coordinates 
            Xh[0]=x;
            Xh[1]=y;
            Xh[2]=1.0f;

            l=F*Xh; // the vector l has a, b, c such that the line equation is ax'+by'+c=0 which is the equation of the epipolar line in I2 where the point X' lies

            // now we draw the line obtained in I2: we take two points:
            // x0=0 (left edge), y0=solution of equation of line
            // x1=I2.width() - 1 (right edge), y1=solution of equation of line

            // if b=0 --> vertical line: we take two points:
            // y0=0 (top edge), x0=solution of equation of line
            // y1=I2.height() - 1 (bottom edge), x1=solution of equation of line

            float x0,x1,y0,y1;
            
            if(fabs(l[1]) < 1e-8f){ // this means it's a vertical line . not enough: because some lines would be deemed vertical when irl they're not. relative !!! ( should be abs b << sqrt a^2 + b^2) 
                y0=0;
                y1=I2.height()-1;
                x0=x1=-l[2]/l[0];
            }
            else{
                x0=0;
                x1=I2.width()-1;
                y0=(-(l[0]*x0)-l[2])/(l[1]);
                y1=(-(l[0]*x1)-l[2])/(l[1]);
            }

            // I2 starts at width of I1 and ends at width of I1 + width of I2 so we take the offset into account when using drawLine() function
            drawLine((int)round(x0+I1.width()), (int)round(y0), (int)round(x1+I1.width()), (int)round(y1), RED);  
  
        }

        else if(x>=I1.width() && x<I1.width()+I2.width() && y>=0 && y<I2.height()){
            cout << "You clicked a point in I2" << endl;
            // user clicked point in I2
            FVector<float,3> l; // this is the epipolar line F^Tx'
            FVector<float,3> Xprimeh;
            Xprimeh[0]=x-I1.width(); // removing offset to go to coordinate system of I2
            Xprimeh[1]=y;
            Xprimeh[2]=1.0f;

            l=transpose(F)*Xprimeh; // the vector l has a, b, c such that the line equation is ax+by+c=0 which is the equation of the epipolar line in I1 where the point X lies

            // now we draw the line obtained in I1: we take two points:
            // x0=0 (left edge), y0=solution of equation of line
            // x1=I1.width() - 1 (right edge), y1=solution of equation of line

            // if b=0 --> vertical line: we take two points:
            // y0=0 (top edge), x0=solution of equation of line
            // y1=I1.height() - 1 (bottom edge), x1=solution of equation of line

            int x0,x1,y0,y1;
            
            if(fabs(l[1]) < 1e-6){
                y0=0;
                y1=I1.height()-1;
                x0=x1=-l[2]/l[0];
            }
            else{
                x0=0;
                x1=I1.width()-1;
                y0=(-(l[0]*x0)-l[2])/(l[1]);
                y1=(-(l[0]*x1)-l[2])/(l[1]);


            }
       
            drawLine((int)round(x0), (int)round(y0), (int)round(x1), (int)round(y1), RED);            
        }
        else{
            cout << "You did not click on neither Image 1 nor Image 2" << endl;
        }
    }
}


int main(int argc, char* argv[])
{
    srand((unsigned int)time(0));

    std::string s1 = argc>1? argv[1]: srcPath("im1.png");
    std::string s2 = argc>2? argv[2]: srcPath("im2.png");

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
    
    std::ifstream file("inliers.txt");

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

    F =HEIV(vinit, Eall, f0);
    // F =FNS(uinit, Eall, Vall);
    // F = Renorm(uinit, Eall, Vall); 
    // F = GaussNewton(uinit, Eall, Vall);

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

    std::cout << "avg epi distance estim: " << avg_epidist_estim << std::endl;

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
