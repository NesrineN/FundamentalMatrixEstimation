#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <cmath>
#include "FNS.h"

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

Mat ComputeP(const Vec& u){
    Mat I = Mat::eye(u.nrow());
    double norm2 = u.qnorm();

    if (norm2 < 1e-12) {
        return I;
    }

    Mat uuT = u * u.t();
    return I - (uuT / norm2);
}

Mat computePseudoInverse(const Mat& A){
    int m=A.nrow();
    int n= A.ncol();
    Mat U(m,m);
    Mat V(n,n);
    Vec S(std::min(m, n));
    A.SVD(U,S,V);

    Mat Sigma_pinv=Mat::zeros(n, m);

    double tol = 1e-10;

    for (int i = 0; i < S.nrow(); i++) {
        if (std::abs(S(i)) > tol) {
            Sigma_pinv(i, i) = 1.0 / S(i);
        }
    }

    Mat A_pinv = V * Sigma_pinv * U.t();

    return A_pinv;

}

Mat GaussNewton(const Vec& u, const Mat& Eall){
    Vec uold=u;
    Vec unew=u;

    for(int i=0; i<100; i++){
        Mat M=ComputeM(uold, Eall);
        Mat L=ComputeL(uold,Eall);
        Mat Pu=ComputeP(uold);

        Mat PuMPu= Pu*M*Pu;

        Mat PuMPu_s=computePseudoInverse(PuMPu);

        Mat ML=M-L;

        // unew=uold- delta u
        unew=uold-(PuMPu_s*(ML*uold));

        // normalizing unew to have unit length
        double norm = std::sqrt(unew.qnorm());
        unew /= norm;

        if (dot(unew, uold) < 0) {
            unew = -unew;
        }

        // because u and -u represent the same fundamental matrix F 
        double d = (unew - uold).qnorm();

        if(d < 1e-6) {break;}

        uold=unew;

    }

    // the solution is unew
    Mat F=Mat::zeros(9);

    for(int i=0;i<3; i++){
        for(int j=0; j<3; j++){
            F(i, j) = unew(i * 3 + j);
        }
    }

    // enforcing rank 2 on F: 

    Mat U(3,3);
    Mat V(3,3);
    Vec S(3);
    F.SVD(U,S,V);

    int minIndex = 0;
        for (int i = 1; i < 3; ++i)
            if (S(i) < S(minIndex))
                minIndex = i;

    S(minIndex)=0;

    Mat Sdiag = Mat::zeros(3,3);
    for(int i=0; i<3; i++) Sdiag(i,i) = S(i);

    F= U*Sdiag*V.t();

    return F;
}