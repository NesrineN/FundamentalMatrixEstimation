#include "libOrsa/libNumerics/matrix.h"
#include <vector>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

struct Point2D
{
    double x;
    double y;
};


// assuming P1= [I|0]
// U and U_prime are in the form of u=w(u,v,1)
double Triangulate(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime){
    // extracting the coordinates from the u and u' vectors
    // assumes U = (u, v, 1)
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
    A.SVD(W,S,V);

    Vec solution = V.col(V.ncol()-1);

    double w_hom = solution(3);

    Mat R=P.copy(0,2,0,2);
    Vec t=P.col(3);

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

        if(z1>0 && z2>0){
            return 1.0;
        }
        else{
            return -1.0;
        }


    }
    else{
        std::cout << "w_hom was zero" << std::endl;
        return -1.0; // point behind camera
    }
}


Mat EstimatePose(const Mat& K1, const Mat& K2, const Mat& F, const std::vector<Point2D>& img1Pts, const std::vector<Point2D>& img2Pts){
    Mat E=K2.t()*F*K1;
    // enforce singular value correction (s,s,0) and normalization of E before!!!
    // E = E / E.norm();  
    // S(2) = 0;
    // S(0) = S(1);

    Mat U(E.nrow(), E.nrow());
    Mat V(E.ncol(), E.ncol());
    Vec S(std::min(E.nrow(), E.ncol()));
    E.SVD(U, S, V);


    Mat W=Mat::zeros(3);
    W(0,1)=-1;
    W(1,0)=1;
    W(2,2)=1;

    Mat R1=U*W*V.t();
    Mat R2=U*W.t()*V.t();

    // should enforce proper rotation
    // if (det(R1) < 0) R1 = -R1;
    // if (det(R2) < 0) R2 = -R2;

    Vec t1=U.col(2);
    Vec t2=-t1;

    // we got the 4 possibilities of poses: R1,t1 - R1, t2 - R1,t2 - R2,t2
    // we do cheirality test to choose one of the 4 candidates
    // should loop over all correspondences and choose the R,t pair that gave maximum count 
    Mat P=Mat::zeros(3,4);
    P.paste(0,0,Mat::eye(3));
    P.paste(0,3,Vec(0,0,0).col(0));

    Mat P1=Mat::zeros(3,4);
    P1.paste(0, 0, R1.copy(0,2,0,2));
    P1.paste(0,3,t1.col(0));
    int chirality1=Triangulate(u, u_prime, P,P1);

    Mat P2=Mat::zeros(3,4);
    P2.paste(0, 0, R1.copy(0,2,0,2));
    P2.paste(0,3,t2.col(0));
    int chirality2=Triangulate(u, u_prime, P,P2);

    Mat P3=Mat::zeros(3,4);
    P3.paste(0, 0, R2.copy(0,2,0,2));
    P3.paste(0,3,t1.col(0));
    int chirality3=Triangulate(u, u_prime, P,P3);

    Mat P4=Mat::zeros(3,4);
    P4.paste(0, 0, R2.copy(0,2,0,2));
    P4.paste(0,3,t2.col(0));
    int chirality4=Triangulate(u, u_prime, P,P4);

    if(chirality1>0){return P1;}
    else if(chirality2>0){return P2;}
    else if(chirality3>0){return P3;}
    else if(chirality4>0){return P4;}
    else {return Mat::zeros(3,4);}

}