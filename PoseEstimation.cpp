#include "libOrsa/libNumerics/matrix.h"
#include <vector>
#include "PoseEstimation.h"

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

// assuming P1= K1[I|0]  P2=K2[R|t]
// U=(u,v,1) and U_prime=(u',v',1) are pixel coordinates 
int Triangulate(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime, const Mat& R, const Vec& t){
    // extracting the coordinates from the u and u' vectors
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
            return 1;
        }
        else{
            return -1;
        }


    }
    else{
        std::cout << "w_hom was zero" << std::endl;
        return -1; // point behind camera
    }
}

double ReprojectionError(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime){
    // extracting the coordinates from the u and u' vectors
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


    if(std::abs(w_hom) > 1e-9){
        Mat X=Mat::zeros(4, 1);
        X(0,0)=solution(0) / w_hom;
        X(1,0)=solution(1) / w_hom;
        X(2,0)=solution(2) / w_hom;
        X(3,0)=1.0;

        // projection onto image1: 
        Mat x1=P*X; // 3x4 * 4x1 --> 3x1
        Mat x2=P_prime*X;

        if (x1(2,0) <= 0 || x2(2,0) <= 0)
        {
           return 0.0; // point is behind the camera i want to skip it
        }

        double u_hat=x1(0,0)/x1(2,0);
        double v_hat=x1(1,0)/x1(2,0);

        double u_hat_p=x2(0,0)/x2(2,0);
        double v_hat_p=x2(1,0)/x2(2,0);

        // std::cout << "Observed Pixel: (" << u << ", " << v << ")" << std::endl;
        // std::cout << "Projected Pixel: (" << u_hat << ", " << v_hat << ")" << std::endl;
        // std::cout << "Homogeneous W: " << x1(2) << std::endl;

        double err1_sq = (u - u_hat) * (u - u_hat) + (v - v_hat) * (v - v_hat);
                         
        double err2_sq = (u_p - u_hat_p) * (u_p - u_hat_p) + (v_p - v_hat_p) * (v_p - v_hat_p);

        // Accumulate error for both projections
        double total_squared_error = (err1_sq + err2_sq);
        return total_squared_error;
    }
    else{
        std::cout << "w_hom was zero!" << std::endl;
        return 0.0;
    }
}

// function that normalizes a 3x3 matrix by converting it to a 9-vector form and dividing it by qnorm and converting it back to matrix form
Mat Normaliza_Mat(const Mat& A){
    Vec A_vec(9);
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            A_vec((i*3) +j)=A(i,j);
        }
    }
    A_vec/=std::sqrt(A_vec.qnorm());

    Mat A_norm=Mat::zeros(3);

    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            A_norm(i,j)=A_vec(i * 3 + j);;
        }
    }

    return A_norm;

}

// assumes P1=[I|0] and returns P2=[R|t] 
Mat EstimatePose(const Mat& K1, const Mat& K2, const Mat& F, const std::vector<Point2D>& img1Pts, const std::vector<Point2D>& img2Pts){
    Mat E=K2.t()*F*K1;

    Mat E_norm=Normaliza_Mat(E); // normalizing E

    Mat U(E_norm.nrow(), E_norm.nrow());
    Mat V(E_norm.ncol(), E_norm.ncol());
    Vec S(std::min(E_norm.nrow(), E_norm.ncol()));
    E_norm.SVD(U, S, V);

    // enforcing singular value correction (s,s,0)  
    double s = (S(0) + S(1)) / 2.0;
    Mat Sigma = Mat::zeros(3);
    Sigma(0,0) = s;
    Sigma(1,1) = s;
    Sigma(2,2) = 0;

    E_norm = U * Sigma * V.t();

    E_norm.SVD(U, S, V);

    Mat W=Mat::zeros(3);
    W(0,1)=-1;
    W(1,0)=1;
    W(2,2)=1;

    Mat R1=U*W*V.t();
    Mat R2=U*W.t()*V.t();

    // U is orthogonal --> vector t is a unit vector --> only direction no magnitude . R is a full, correctly-scaled rotation, but t is only a unit direction
    Vec t1=U.col(2);
    Vec t2=-t1;

    // enforcing proper rotation
    if(R1.det()<0){
        R1=-R1;
    }

    if(R2.det()<0){
        R2=-R2;
    } 

    // we got the 4 possibilities of poses: R1,t1 - R1, t2 - R2,t1 - R2,t2

    Mat P1=Mat::zeros(3,4);
    P1.paste(0,0,Mat::eye(3));
    P1.paste(0,3,Vec(0,0,0).col(0));

    Mat P2_1=Mat::zeros(3,4);
    P2_1.paste(0, 0, R1.copy(0,2,0,2));
    P2_1.paste(0,3,t1.col(0));

    Mat P2_2=Mat::zeros(3,4);
    P2_2.paste(0, 0, R1.copy(0,2,0,2));
    P2_2.paste(0,3,t2.col(0));

    Mat P2_3=Mat::zeros(3,4);
    P2_3.paste(0, 0, R2.copy(0,2,0,2));
    P2_3.paste(0,3,t1.col(0));

    Mat P2_4=Mat::zeros(3,4);
    P2_4.paste(0, 0, R2.copy(0,2,0,2));
    P2_4.paste(0,3,t2.col(0));


    // we loop through the point correspondences and for each point we run the triangulation to compute the cheirality with each of the four poses:
    int total_1=0, total_2=0, total_3=0, total_4=0;

    for(size_t i=0; i<img1Pts.size(); i++){
        Vec u(3);
        Vec u_prime(3);

        u(0)=img1Pts[i].x;
        u(1)=img1Pts[i].y;
        u(2)=1.0;

        u_prime(0)=img2Pts[i].x;
        u_prime(1)=img2Pts[i].y;
        u_prime(2)=1.0;

        if (Triangulate(u,u_prime,K1*P1,K2*P2_1, R1, t1) > 0)total_1++;
        if (Triangulate(u,u_prime,K1*P1,K2*P2_2, R1, t2) > 0)total_2++;
        if (Triangulate(u,u_prime,K1*P1,K2*P2_3, R2, t1) > 0)total_3++;
        if (Triangulate(u,u_prime,K1*P1,K2*P2_4, R2, t2) > 0)total_4++;
    }

    std::cout << "Totals: " << std::endl;
    std::cout << total_1 << " "
          << total_2 << " "
          << total_3 << " "
          << total_4 << std::endl;

    int maxTotal = total_1;
    Mat bestP = P2_1;

    if (total_2 > maxTotal) {
        maxTotal = total_2;
        bestP = P2_2;
    }

    if (total_3 > maxTotal) {
        maxTotal = total_3;
        bestP = P2_3;
    }

    if (total_4 > maxTotal) {
        maxTotal = total_4;
        bestP = P2_4;
    }

    return bestP;

}