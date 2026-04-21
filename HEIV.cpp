#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

Mat ComputeV0Z(const Vec& E){
    
    Mat V0=Mat::zeros(8);

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

    // R1
    V0(1,0)= xp*yp;
    V0(1,1)= x*x + yp*yp;
    V0(1,2)= f0 * yp;
    V0(1,3)= 0;
    V0(1,4)= x*y;
    V0(1,5)=0;
    V0(1,6)=0;
    V0(1,7)=f0*x;

    // R2
    V0(2,0)= f0*xp;
    V0(2,1)= f0*yp;
    V0(2,2)= f0 * f0;
    V0(2,3)= 0;
    V0(2,4)= 0;
    V0(2,5)=0;
    V0(2,6)=0;
    V0(2,7)=0;

    // R3
    V0(3,0)= x*y;
    V0(3,1)= 0;
    V0(3,2)= 0;
    V0(3,3)= y*y + xp*xp;
    V0(3,4)= xp*yp;
    V0(3,5)= f0*xp;
    V0(3,6)= f0*y;
    V0(3,7)=0;

    // R4
    V0(4,0)= 0;
    V0(4,1)= x*y;
    V0(4,2)= 0;
    V0(4,3)= xp*yp;
    V0(4,4)= y*y + yp*yp;
    V0(4,5)= f0*yp;
    V0(4,6)= 0;
    V0(4,7)=f0*y;

    // R5
    V0(5,0)= 0;
    V0(5,1)= 0;
    V0(5,2)= 0;
    V0(5,3)= f0*xp;
    V0(5,4)= f0*yp;
    V0(5,5)= f0*f0;
    V0(5,6)= 0;
    V0(5,7)=0;

    // R6
    V0(6,0)= f0*x;
    V0(6,1)= 0;
    V0(6,2)= 0;
    V0(6,3)= f0*y;
    V0(6,4)= 0;
    V0(6,5)= 0;
    V0(6,6)= f0*f0;
    V0(6,7)=0;

    // R7
    V0(7,0)= 0;
    V0(7,1)= f0*x;
    V0(7,2)= 0;
    V0(7,3)= 0;
    V0(7,4)= f0*y;
    V0(7,5)= 0;
    V0(7,6)= 0;
    V0(7,7)= f0*f0;

    return V0;

}

Mat ComputeMTilde(const Vec& v, const Mat& Zall, const Mat& Ztall) {
    int n=Zall.ncol();
    Mat M=Mat::zeros(8);

    for(int i=0; i<n; i++){
        Vec Z=Zall.col(i);
        Vec Zt=Ztall.col(i);

        Mat V0Z=ComputeV0Z(Z); // remove this later and compute V0 for each point correspondence once and store it in a vector of matrices for efficiency 
        Vec V0Zv= V0Z*v;
        double vV0Zv= dot(v,V0Zv);
        
        Mat S= Zt*Zt.t();
        S=S/vV0Zv;
        M=M+S;
    }

    return M;
}

Mat ComputeLTilde(const Vec& v, const Mat& Zall, const Mat& Ztall) {
    int n=Zall.ncol();
    Mat L=Mat::zeros(8);

    for(int i=0; i<n; i++){
        Vec Z=Zall.col(i);
        Vec Zt=Ztall.col(i);
        
        Mat V0Z=ComputeV0Z(Z);
        Vec V0Zv= V0Z*v;

        double vV0Zv= dot(v,V0Zv);
        double vZt=dot(v,Zt);
        Mat S= V0Z;

        S=S*((vZt)*(vZt));
        S=S/((vV0Zv)*(vV0Zv));
        L=L+S;
    }

    return L;
}

Vec solveGeneralizedEigen(const Mat& Mt, const Mat& Lt){

    Eigen::MatrixXd eigenM(Mt.nrow(), Mt.ncol());

    for (int i = 0; i < Mt.nrow(); ++i) {
        for (int j = 0; j < Mt.ncol(); ++j) {
            eigenM(i, j) = Mt(i, j);
        }
    }

    Eigen::MatrixXd eigenL(Lt.nrow(), Lt.ncol());

    for (int i = 0; i < Lt.nrow(); ++i) {
        for (int j = 0; j < Lt.ncol(); ++j) {
            eigenL(i, j) = Lt(i, j);
        }
    }

    Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> solver(eigenM, eigenL);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed");
    }

    auto eigenvalues = solver.eigenvalues();
    auto eigenvectors = solver.eigenvectors();

    int bestIndex = 0;
    double bestDist = std::abs(eigenvalues[0] - std::complex<double>(1.0, 0.0));

    for (int i = 1; i < eigenvalues.size(); ++i) {
        double dist = std::abs(eigenvalues[i] - std::complex<double>(1.0, 0.0));
        if (dist < bestDist) {
            bestDist = dist;
            bestIndex = i;
        }
    }

    Eigen::VectorXcd v = eigenvectors.col(bestIndex);
    v.normalize();

    Vec vnew(v.size());

    for (int i = 0; i < v.size(); ++i) {
        vnew(i) = v[i].real();
    }

    return vnew;

}

// takes in the initial guess of v and Zall and Ztall and applies the HEIV algorithm to estimate the fundamental matrix F
Mat HEIV(const Vec& v, const Mat& Zall, const Mat& Ztall, Vec Zbar, double f0){
    
    Vec vold=v;
    Vec vnew=v;

    for(int i=0; i<100; i++){
        Mat Mt=ComputeMTilde(vold, Zall, Ztall);
        Mat Lt=ComputeLTilde(vold, Zall, Ztall);

        // standard eigen value problem: 
        vnew = solveGeneralizedEigen(Mt, Lt);

        if((vnew-vold).qnorm()<1e-10){break;}

        vold=vnew;

    }

    // the solution is vnew
    double F33= - (dot(vnew,Zbar)) / (f0*f0);
    Vec unew(vnew(0), vnew(1), vnew(2), vnew(3), vnew(4), vnew(5), vnew(6), vnew(7), F33);

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