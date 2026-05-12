#include "libOrsa/libNumerics/matrix.h"
#include "HEIV.h"
#include <iostream> 
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

void printMat(const Mat& A){
    std::cout << "Matrix: " << std::endl;
    for(int i=0;i<A.nrow(); i++){
        for(int j =0; j<A.ncol(); j++){
            std::cout << A(i,j) << " "; 
        }
        std::cout << "" << std::endl;
        std::cout << "" << std::endl;
    }
}

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

Mat ComputeNTBt(const Mat& Eall, double f0, const std::vector<Mat>& Vall) {
    
    int n=Eall.ncol();
    Mat NTBt=Mat::zeros(8);

    for(int i=0; i<n; i++){
        Vec E=Eall.col(i);
        Mat V0z=Vall[i].copy(0,7,0,7);
        // Mat V0z=ComputeV0Z(E, f0);
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

Vec Taubin(const Mat& Eall, double f0, const std::vector<Mat>& Vall){

    Vec Zbar(8);
    for(int i=0; i<8; i++){
        Zbar(i)=0;
    }
    for(int i=0; i<Eall.ncol(); i++){
        Vec E8=Eall.col(i).copy(0,7);
        Zbar=Zbar+E8;
    }

    Zbar=Zbar/Eall.ncol();

    Mat MLSt=ComputeMLSt(Eall, Zbar);
    Mat NTBt=ComputeNTBt(Eall, f0, Vall);

    Vec v=SolveGeneralizedEigen(MLSt, NTBt); // chooses vector with smallest lambda.
    v/=std::sqrt(v.qnorm());
    double F33=-(dot(v,Zbar))/(f0*f0);

    Vec u(9);

    for (int i = 0; i < 8; ++i)
        u(i) = v(i);
    
    u/=std::sqrt(u.qnorm());
    
    u(8) = F33;

    Mat F=Mat::zeros(3);
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            F(i, j) = u(i * 3 + j);
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

    Vec utaubin(9);
    // for(int i = 0; i < 9; i++) utaubin(i) = F(i/3, i%3);
    utaubin(0)=F(0,0);
    utaubin(1)=F(0,1);
    utaubin(2)=F(0,2);
    utaubin(3)=F(1,0);
    utaubin(4)=F(1,1);
    utaubin(5)=F(1,2);
    utaubin(6)=F(2,0);
    utaubin(7)=F(2,1);
    utaubin(8)=F(2,2);

    utaubin/=std::sqrt(utaubin.qnorm());

    return utaubin;
}