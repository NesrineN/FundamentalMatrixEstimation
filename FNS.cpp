#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

// Eall is a matrix that stores E for each point correspondence. each E is a 9x1 vector and so Eall is a 9xn matrix where each col corresponds to a point correspondence and so we have n columns in total --> n total point correspondences
Mat ComputeM(const Vec& u, const Mat& Eall, const std::vector<Mat>& Vall) {
    int n=Eall.ncol();
    Mat M=Mat::zeros(9);

    for(int i=0; i<n; i++){
        Vec E=Eall.col(i);
        Mat V0=Vall[i];
        Vec V0u= V0*u;
        double uV0u= dot(u,V0u);
        Mat S= E*E.t(); 
        if(std::abs(uV0u) < 1e-12) continue;
        S=S/uV0u;
        M=M+S;
    }

    return M;
}

Mat ComputeL(const Vec& u, const Mat& Eall, const std::vector<Mat>& Vall) {
    int n=Eall.ncol();
    Mat L=Mat::zeros(9);

    for(int i=0; i<n; i++){
        Vec E=Eall.col(i);
        Mat V0=Vall[i];
        Vec V0u= V0*u;
        double uV0u= dot(u,V0u);
        if(std::abs(uV0u) < 1e-12) continue;
        double uE=dot(u,E);
        
        Mat S= V0;

        S=S*((uE)*(uE));
        S=S/((uV0u)*(uV0u));
        L=L+S;
    }

    return L;
}

// returns the vector u corresponding to the smallest eigen value of the matrix A (M-L) (modified FNS)
// M-L is symmetric --> SelfAdjointEigenSolver should be more stable numerically   
Vec SolveEigen(const Mat& A){

    Eigen::MatrixXd eigenA(A.nrow(), A.ncol());

    for (int i = 0; i < A.nrow(); ++i) {
        for (int j = 0; j < A.ncol(); ++j) {
            eigenA(i, j) = A(i, j);
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(eigenA);

    if (es.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed");
    }

    //Pick the algebraically smallest eigenvalue
    // Because they are sorted, this is always index 0.
    Eigen::VectorXd u = es.eigenvectors().col(0);

    // debug
    // std::cout << "Eigen value:" << es.eigenvalues()(0) << std::endl;


    u.normalize();

    int size=u.size();
    Vec unew(size);

    for (int i = 0; i < u.size(); ++i) {
        unew(i) = u(i);
    }

    return unew;
}

// takes in the initial guess of u, V0, and Eall and applies the FNS algorithm and returns the final Fundamental Matrix F
Mat FNS(const Vec& u, const Mat& Eall, const std::vector<Mat>& Vall){
    
    Vec uold=u;
    Vec unew=u;

    for(int i=0; i<100; i++){
        Mat M=ComputeM(uold, Eall, Vall);
        Mat L=ComputeL(uold,Eall, Vall);

        Mat ML=M-L;
        unew = SolveEigen(ML);

        // because u and -u represent the same fundamental matrix F 
        double d1 = (unew - uold).qnorm();
        double d2 = (unew + uold).qnorm();

        uold=unew;

        if(std::min(d1,d2) < 1e-8) {break;}

    }

    // normalizing unew to have unit length just in case
    double norm = std::sqrt(unew.qnorm());
    unew /= norm;
    
    // the solution is unew
    Mat F=Mat::zeros(3);

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

// CFNS


