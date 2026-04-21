#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <cmath>
#include "FNS.h"

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

Mat ComputeN(const Vec& u, const Mat& Eall) {
    int n=Eall.ncol();
    Mat N=Mat::zeros(9);

    for(int i=0; i<n; i++){
        Vec E=Eall.col(i);
        Mat V0=ComputeV0(E);
        Vec V0u= V0*u;
        double uV0u= dot(u,V0u);
        Mat S= V0;
        S=S/uV0u;
        N=N+S;
    }

    return N;
}

// takes in the initial guess of u, V0, and Eall and applies the FNS algorithm and returns the final Fundamental Matrix F
Mat Renorm(const Vec& u, const Mat& Eall){
    
    Vec uold=u;
    Vec unew=u;
    double c=0;
    double lambda=1e19;

    for(int i=0; i<100; i++){
        Mat M=ComputeM(uold, Eall);
        Mat N=ComputeN(uold,Eall);

        Mat McN=M-(c*N);

        Mat U(9,9);
        Mat V(9,9);
        Vec S(9);
        McN.SVD(U,S,V);

        int minIndex = 0;
        for (int i = 1; i < 9; ++i)
            if (S(i) < S(minIndex))
                minIndex = i;

        unew = V.col(minIndex);
        lambda= S(minIndex);

        if(lambda<1e-9){break;}

        Vec Nunew= N*unew;
        double unewNunew=dot(unew, Nunew);
        c=c+(lambda/unewNunew);
        uold=unew;

    }
    // the solution is unew

    // normalizing unew to have unit length
    double norm = std::sqrt(unew.qnorm());
    unew /= norm;


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