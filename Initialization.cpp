#include "libOrsa/libNumerics/matrix.h"
#include "HEIV.h"
#include <iostream> 
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

// returns a random initialization of the vector u where each component is a random nbr sampled from a gaussian distribution of mean 0 and sdv 1.
Vec RandomInit(){
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    Vec u(9);
    for (int i = 0; i < 9; ++i) {
        u(i) = dist(gen);
    }

    double norm = std::sqrt(u.qnorm());
    u /= norm;

    return u;
}

Mat ComputeMLS(const Mat& Eall) {
    
    int n=Eall.ncol();
    Mat M_LS=Mat::zeros(9);
    for(int i=0; i<n; i++){
        Vec E=Eall.col(i);
        Mat S= E*E.t();
        M_LS=M_LS+S;
    }

    return M_LS;
}

Mat ComputeMLSt(const Mat& Eall, const Vec& Zbar) {
    
    int n=Eall.ncol();
    Mat M_LSt=Mat::zeros(8);

    for(int i=0; i<n; i++){
        Vec Z=Eall.col(i).copy(0,7);
        Vec Zt=Z-Zbar;
        Mat S= Zt*Zt.t();
        M_LSt=M_LSt+S;
    }

    return M_LSt;
}

Mat ComputeNTBt(const Mat& Eall) {
    
    int n=Eall.ncol();
    Mat NTBt=Mat::zeros(8);

    for(int i=0; i<n; i++){
        Vec E=Eall.col(i);
        Mat V0z=ComputeV0Z(E);
        Mat S=V0z;
        NTBt=NTBt+S;
    }

    return NTBt;
}

Vec LeastSquares(const Mat& Eall){
    
    Mat M_LS=ComputeMLS(Eall);
    Mat U(9,9);
    Mat V(9,9);
    Vec S(9);
    M_LS.SVD(U,S,V);

    int minIndex = 0;
    for (int i = 1; i < 9; ++i)
        if (S(i) < S(minIndex))
            minIndex = i;

    Vec u = V.col(minIndex);

    double norm = std::sqrt(u.qnorm());
    u /= norm;

    return u;
}

Vec Taubin(const Mat& Eall){

    Vec firstcol=Eall.col(0);
    double f0=std::sqrt(firstcol(8));

    Vec Zbar(8);
    for(int i=0; i<8; i++){
        Zbar(i)=0
    }
    for(int i=0; i<Eall.ncol(); i++){
        Zbar=Zbar+(Eall.col(i).copy(0,7));
    }
    Zbar=Zbar/Eall.ncol();
 
    Mat MLSt=ComputeMLSt(Eall, Zbar);
    Mat NTBt=ComputeNTBt(Eall);

    Vec v=solveGeneralizedEigen(MLSt, NTBt); // chooses vector with smallest lambda.
    double F33=-(dot(v,Zbar))/(f0*f0);

    Vec u(9);

    for (int i = 0; i < 8; ++i)
        u(i) = v(i);

    u(8) = F33;

    double norm = std::sqrt(u.qnorm());
    u /= norm;

    return u;
}