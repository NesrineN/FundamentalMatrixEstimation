#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

Mat ComputeV0(const Vec& E){
    
    Mat V0=Mat::zeros(9);

    double f0=std::sqrt(E(8));
    double x=E(2)/f0;
    double y=E(5)/f0;
    double xp=E(6)/f0;
    double yp=E(7)/f0;

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

// Eall is a matrix that stores E for each point correspondence. each E is a 9x1 vector and so Eall is a 9xn matrix where each col corresponds to a point correspondence and so we have n columns in total --> n total point correspondences
Mat ComputeM(const Vec& u, const Mat& Eall) {
    int n=Eall.ncol();
    Mat M=Mat::zeros(9);

    for(int i=0; i<n; i++){
        Vec E=Eall.col(i);
        Mat V0=ComputeV0(E); // remove this later and compute V0 for each point correspondence once and store it in a vector of matrices for efficiency 
        Vec V0u= V0*u;
        double uV0u= dot(u,V0u);
        Mat S= E*E.t();
        if(std::abs(uV0u) < 1e-15){
            uV0u+=1e-9;
        }
        S=S/uV0u;
        M=M+S;
    }

    return M;
}

Mat ComputeL(const Vec& u, const Mat& Eall) {
    int n=Eall.ncol();
    Mat L=Mat::zeros(9);

    for(int i=0; i<n; i++){
        Vec E=Eall.col(i);
        Mat V0=ComputeV0(E);
        Vec V0u= V0*u;
        double uV0u= dot(u,V0u);
        if(std::abs(uV0u) < 1e-15){
            uV0u+=1e-9;
        }
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
    u.normalize();

    Vec unew(u.size());

    for (int i = 0; i < u.size(); ++i) {
        unew(i) = u(i);
    }

    return unew;
}

// takes in the initial guess of u, V0, and Eall and applies the FNS algorithm and returns the final Fundamental Matrix F
Mat FNS(const Vec& u, const Mat& Eall){
    
    Vec uold=u;
    Vec unew=u;

    for(int i=0; i<100; i++){
        Mat M=ComputeM(uold, Eall);
        Mat L=ComputeL(uold,Eall);

        Mat ML=M-L;
        unew = SolveEigen(ML);

        // because u and -u represent the same fundamental matrix F 
        double d1 = (unew - uold).qnorm();
        double d2 = (unew + uold).qnorm();

        if(std::min(d1,d2) < 1e-10) {break;}

        uold=unew;

    }

    // normalizing unew to have unit length just in case
    double norm = std::sqrt(unew.qnorm());
    unew /= norm;
    
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

// CFNS is when we force det(F)=0 during the iterations and not as a post-processing step
// Mat CFNS(const Vec& u, const Mat& Eall){
    
//     Vec uold=u;
//     Vec unew=u;

//     for(int i=0; i<100; i++){
//         Mat M=ComputeM(uold, Eall);
//         Mat L=ComputeL(uold,Eall);

//         Mat ML=M-L;

//         Mat U(9,9);
//         Mat V(9,9);
//         Vec S(9);
//         ML.SVD(U,S,V);

//         int minIndex = 0;
//         for (int i = 1; i < 9; ++i)
//             if (S(i) < S(minIndex))
//                 minIndex = i;

//         unew = V.col(minIndex);

//         if((unew-uold).qnorm()<1e-10){break;}

//         uold=unew;

//     }
//     // the solution is unew
//     Mat F=Mat::zeros(9);

//     for(int i=0;i<3; i++){
//         for(int j=0; j<3; j++){
//             F(i, j) = unew(i * 3 + j);
//         }
//     }

//     // enforcing rank 2 on F: 

//     Mat U(3,3);
//     Mat V(3,3);
//     Vec S(3);
//     F.SVD(U,S,V);

//     int minIndex = 0;
//         for (int i = 1; i < 3; ++i)
//             if (S(i) < S(minIndex))
//                 minIndex = i;

//     S(minIndex)=0;

//     Mat Sdiag = Mat::zeros(3,3);
//     for(int i=0; i<3; i++) Sdiag(i,i) = S(i);

//     F= U*Sdiag*V.t();

//     return F;
// }


