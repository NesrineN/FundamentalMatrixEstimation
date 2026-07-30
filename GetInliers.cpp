#include <iostream> 
#include <cmath>
#include <vector>
#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "./Imagine/Features.h"
#include "Match.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include "libOrsa/libNumerics/matrix.h"
#include "ORSAWrapper.h"
#include "PoseEstimation.h"

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

using namespace Imagine;
using namespace std;

// struct Match {
//     double x1, y1, x2, y2;
// };

void printM_here(const Mat& A){
    std::cout << "Matrix: " << std::endl;
    for(int i=0;i<A.nrow(); i++){
        for(int j =0; j<A.ncol(); j++){
            std::cout << A(i,j) << " "; 
        }
        std::cout << "" << std::endl;
        std::cout << "" << std::endl;
    }
}

static const double BETA = 0.01f; // Probability of failure

// UNDISTORTING THE PIXELS:
// Undistorts a single pixel point (u,v) given camera intrinsics and distortion coefficients.
// Returns the corresponding *undistorted* pixel coordinates, still in pixel units.
Point2D undistortPoint(double u, double v,
                        double fx, double fy, double cx, double cy,
                        double k1, double k2, double p1, double p2, double k3) {
    // Step 1: convert to normalized (distorted) coordinates
    double x = (u - cx) / fx;
    double y = (v - cy) / fy;

    double x0 = x, y0 = y;

    // Step 2: iteratively solve for the undistorted normalized coordinates
    // (standard fixed-point iteration, same approach OpenCV's undistortPoints uses)
    for (int iter = 0; iter < 10; iter++) {
        double r2 = x*x + y*y;
        double r4 = r2*r2;
        double r6 = r4*r2;
        double radial = 1.0 + k1*r2 + k2*r4 + k3*r6;

        double dx = 2*p1*x*y + p2*(r2 + 2*x*x);
        double dy = p1*(r2 + 2*y*y) + 2*p2*x*y;

        x = (x0 - dx) / radial;
        y = (y0 - dy) / radial;
    }

    // Step 3: convert back to pixel coordinates using the SAME K
    // (so the result is still in pixel space, just distortion-corrected)
    Point2D result;
    result.x = x * fx + cx;
    result.y = y * fy + cy;
    return result;
}

void undistortMatches(std::vector<SiftMatch>& matches,
                       double fx, double fy, double cx, double cy,
                       double k1, double k2, double p1, double p2, double k3) {
    for (auto& m : matches) {
        Point2D p1u = undistortPoint(m.x1, m.y1, fx, fy, cx, cy, k1, k2, p1, p2, k3);
        Point2D p2u = undistortPoint(m.x2, m.y2, fx, fy, cx, cy, k1, k2, p1, p2, k3);
        m.x1 = p1u.x; m.y1 = p1u.y;
        m.x2 = p2u.x; m.y2 = p2u.y;
    }
}

// first we will implement a function that takes a couple of images and extracts the matching points from them: 
// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<SiftMatch>& matches) {

    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    // drawFeatures(feats1, Coords<2>(0,0));
    // std::cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    // drawFeatures(feats2, Coords<2>(I1.width(),0));
    // std::cout << " Im2: " << feats2.size() << flush;

    const double RATIO_THRESH = 0.6; // Lowe ratio 0.6

    // const double MAX_DISTANCE = 100.0*100.0;

    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];

        // added:
        double best = std::numeric_limits<double>::max();
        double second = std::numeric_limits<double>::max();
        int bestJ = -1;

        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < best) {
                second = best;
                best = d;
                bestJ = (int)j;
            } else if (d<second) {
                second=d;
            }
        }

        if(bestJ<0) continue;

        // ratio test: best match must be clearly better than the runner-up
        // (squared distances, so compare best < ratio^2 * second)
        if (best < RATIO_THRESH * RATIO_THRESH * second) {
            SiftMatch m;
            m.x1 = f1.pos.x();
            m.y1 = f1.pos.y();
            m.x2 = feats2[bestJ].pos.x();
            m.y2 = feats2[bestJ].pos.y();
            matches.push_back(m);
        }
    }
}

void removeDuplicateMatches(std::vector<SiftMatch>& matches, double eps = 1e-6) {
    // sort by (x1,y1,x2,y2) so duplicates end up adjacent
    std::sort(matches.begin(), matches.end(), [](const SiftMatch& a, const SiftMatch& b) {
        if (std::abs(a.x1 - b.x1) > 1e-9) return a.x1 < b.x1;
        if (std::abs(a.y1 - b.y1) > 1e-9) return a.y1 < b.y1;
        if (std::abs(a.x2 - b.x2) > 1e-9) return a.x2 < b.x2;
        return a.y2 < b.y2;
    });

    auto isDuplicate = [eps](const SiftMatch& a, const SiftMatch& b) {
        return std::abs(a.x1 - b.x1) < eps && std::abs(a.y1 - b.y1) < eps &&
               std::abs(a.x2 - b.x2) < eps && std::abs(a.y2 - b.y2) < eps;
    };

    auto last = std::unique(matches.begin(), matches.end(), isDuplicate);
    matches.erase(last, matches.end());
}

// Function for computing the Normalization Matrices that we will use to normalize the matches
// Isotropic scaling taken from https://www.r-5.org/files/books/computers/algo-list/image-processing/vision/Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision-EN.pdf (P.107)
// "The points are translated so that their centroid is at the origin".
// "The points are then scaled so that the average distance from the origin is equal to √2".
// "This transformation is applied to each of the two images independently".
vector<FMatrix<double,3,3>> compute_N(vector<SiftMatch>& matches){
    
    vector<FMatrix<double,3,3>> N_list;
    FMatrix<double,3,3> N1; // the Normalization matrix for Image 1
    FMatrix<double,3,3> N2; // the Normalization matrix for Image 2

    // we will calculate the centroid of points of each image, then the distance from all points of an image to its centroid, get the scale and fill the matrix with it and the centroid
    // the aim is to translate the points so their centroid is at the origin and to scale the points so that the average distance from the origin is sqrt(2). 

    // for I1:
    double xbar1=0.f;
    double ybar1=0.f;
    for(int j=0;j<matches.size();j++){
        xbar1=xbar1+matches[j].x1;
        ybar1=ybar1+matches[j].y1;
    }
    xbar1/=(double)matches.size(); // centroid
    ybar1/=(double)matches.size(); // centroid

    double distance1=0.f;

    for(int ii=0;ii<matches.size();ii++){
        distance1=distance1+(sqrt(pow((matches[ii].x1-xbar1),2)+pow((matches[ii].y1-ybar1),2)));
    }

    distance1/=(double)matches.size(); // distance of points from the centroid
    if(distance1 < 1e-8f) distance1 = 1.0f; // to avoid dividing by zero in case distance is very close to zero
    double s1=sqrt(2.0f)/distance1; // scale to normalize by
    
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
    double xbar2=0.f;
    double ybar2=0.f;
    for(int id=0;id<matches.size();id++){
        xbar2=xbar2+matches[id].x2;
        ybar2=ybar2+matches[id].y2;
    }
    xbar2/=(double)matches.size();
    ybar2/=(double)matches.size();
    double distance2=0.f;
    for(int ind=0;ind<matches.size();ind++){
        distance2=distance2+(sqrt(pow((matches[ind].x2-xbar2),2)+pow((matches[ind].y2-ybar2),2)));
    }
    distance2/=(double)matches.size();
    if(distance2 < 1e-8f) distance2 = 1.0f;    
    double s2=sqrt(2.0f)/distance2;

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
vector<SiftMatch> normalize_matches(FMatrix<double,3,3> N1, FMatrix<double,3,3> N2, vector<SiftMatch>& matches){
    
    vector<SiftMatch> subset_normalized;
    for(int match_id=0;match_id<matches.size(); match_id++){
        SiftMatch m=matches[match_id];
        SiftMatch m_normalized;

        FVector<double,3> X1h, X2h; // point correspondences in a match in homogeneous system
        X1h[0]= m.x1;  X1h[1]= m.y1;  X1h[2]= 1.0f;
        X2h[0]= m.x2;  X2h[1]= m.y2;  X2h[2]= 1.0f;

        // we normalize both
        FVector<double,3> X1n= N1 * X1h;
        FVector<double,3> X2n= N2 * X2h;

        // we assign to m_normalized after returning to euclidean 2d:
        m_normalized.x1= X1n[0] / X1n[2];
        m_normalized.y1= X1n[1] / X1n[2];

        m_normalized.x2= X2n[0] / X2n[2];
        m_normalized.y2= X2n[1] / X2n[2];

        subset_normalized.push_back(m_normalized);
    }

    return subset_normalized;

}

// Function to classify a match in matches as an inlier/outlier based on an Fcandidate. We check the Sampson error/distance : https://scispace.com/pdf/a-robust-method-for-estimating-the-fundamental-matrix-5giw378zat.pdf
// for each correspondence, we calculate the epipolar constraint x'^T . F. x , and the epipolar lines F.x and F^T.x'
// error is d^2= epipolar constraint ^2 / a^2 + b^2 + a'^2 + b'^2   where a and b are the first 2 coefficients of Fx and a' and b' are the first 2 coefficients of F^T.x'
// finally, we check if d is <= distMax (threshold) , if yes , we mark as inlier and return the index of the inlier
vector<int> mark_inliers(FMatrix<double,3,3>& Fcandid, vector<SiftMatch>& matches, double distMax){
        
        vector<int> Inliers; // has the indices of the matches considered inliers.
        for(int id=0; id<matches.size();id++){
            SiftMatch m=matches[id];

            FVector<double,3> X1h, X2h; // point correspondences in a match in homogeneous system
            X1h[0]= m.x1;  X1h[1]= m.y1;  X1h[2]= 1.0f;
            X2h[0]= m.x2;  X2h[1]= m.y2;  X2h[2]= 1.0f;

            double l[3]; // this is the epipolar line F.xi
            for (int j = 0; j < 3; j++){
                l[j] = Fcandid(j,0)*X1h[0] + Fcandid(j,1)*X1h[1] + Fcandid(j,2)*X1h[2];
            }

            double l2[3]; // this is the epipolar line F^T.x'i
            FMatrix<double,3,3> Ftcandid = transpose(Fcandid);
            for (int j = 0; j < 3; j++){
                l2[j] = Ftcandid(j,0)*X2h[0] + Ftcandid(j,1)*X2h[1] + Ftcandid(j,2)*X2h[2];
            }            

            // now we calculate the sampson distance
            
            // we split the calculation of the epipolar constraint for better ease
            // Step 1: Fx = F * x
            double Fx[3];
            for (int i = 0; i < 3; ++i) {
                Fx[i] = 0.0f;
                for (int j = 0; j < 3; ++j) {
                    Fx[i] += Fcandid(i, j) * X1h[j];  
                }
            }

            // Step 2: s = x'^T * Fx
            double s = 0.0;
            for (int i = 0; i < 3; ++i) {
                s += X2h[i] * Fx[i];
            }

            // Step 3: numerator
            double numerator = s * s;


            double denom=l[0]*l[0] + l[1]*l[1] + l2[0]*l2[0] + l2[1]*l2[1];

            if(denom<1e-8f){denom=1e-8f;} // to make sure we don't divide by zero
            
            double di2 = numerator/denom; 
            double di=sqrt(di2);
            
            if(di<=distMax){
                // m is an inlier, we add it to the temporary vector Inliers which stores the inliers' indices 
                Inliers.push_back((int)id);
            }
        }
        return Inliers;
}

// Function that calculates the Fundamental Matrix using correspondences in matches vector. It returns the F computed and the rank of the matrix A (done to skip samples in RANSAC that produced a degenerate A)
FMatrix<double,3,3> eightpointalgo(vector<SiftMatch>& matches, const FMatrix<double,3,3>& N1, const FMatrix<double,3,3>& N2){
    
    FMatrix<double,3,3> Fcandid; // to be returned 

    int n=matches.size(); // number of matches we have in our case
    // we need AT LEAST 8 points
    if(n < 8){
    Fcandid.fill(0.0f);
    return Fcandid;
    }

    // STEP 1: We normalize the matches

    // FMatrix<float,3,3> N1 ;
    // FMatrix<float,3,3> N2;

    // vector<FMatrix<float,3,3>> N_list;
    // N_list=compute_N(matches);
    // N1=N_list[0];
    // N2=N_list[1];

    vector<SiftMatch> subset_normalized;
    subset_normalized=normalize_matches(N1, N2, matches);

    // STEP 2: We create matrix A and get its SVD

    // A was created using the epipolar constraint x'^T.F.x=0 

    Matrix<double> A(n,9);

    for(int j=0;j<n;j++){
        A(j,0)=subset_normalized[j].x2*subset_normalized[j].x1;
        A(j,1)=subset_normalized[j].x2*subset_normalized[j].y1;
        A(j,2)=subset_normalized[j].x2;
        A(j,3)=subset_normalized[j].y2*subset_normalized[j].x1;
        A(j,4)=subset_normalized[j].y2*subset_normalized[j].y1;
        A(j,5)=subset_normalized[j].y2;
        A(j,6)=subset_normalized[j].x1;
        A(j,7)=subset_normalized[j].y1;
        A(j,8)=1;
    }

    // computing SVD of A
    Matrix<double> U(n,n);
    Matrix<double> V(9,9);
    int minimum=min(n,9);
    Vector<double>S(minimum);
    svd(A, U, S, V);

    int rank = 0;
    for (int si = 0; si < S.size(); si++) {
        if (S[si] > 1e-6f) rank++;
    }

    // if rank of A is less than 8 that means it's degenerate so we return an empty F
    if(rank<8){
        Fcandid.fill(0.0f);
        return Fcandid;
    }

    Vector<double>f(9);
    f=V.getRow(V.nrow() - 1); // this is the column V9 which is the solution to Af=0 (the svd function returns V as V^T that's why we took row instead of column)

    // Reshaping the vector f into a matrix Fncandid (Fnormalized-candidate)
    FMatrix<double,3,3> Fncandid;
    for (int id=0; id<3; id++){
        for (int j=0; j<3; j++){
            Fncandid(id,j)= f[id*3 + j];
        }
    }

    // STEP 3: we set singular value in 3rd column to 0 and recompute Fnormalized-candidate
    FMatrix<double,3,3> Uf;
    FMatrix<double,3,3> Vf;
    FVector<double,3> Sf;
    svd(Fncandid, Uf, Sf, Vf);
    Sf[2] = 0;

    // changing the Sf vector to a 3x3 matrix:
    FMatrix<double,3,3> Sfm; 
    Sfm.fill(0.0f);
    for(int id=0;id<3;id++){
        for(int j=0;j<3;j++){
            if(id==j){
                Sfm(id,id)=Sf[id];
            }
        }
    }

    Fncandid=Uf*Sfm*Vf;

    // debugging:
    // Mat Fncandid_mat=Mat::zeros(3);
    // for (int i = 0; i < 3; i++)
    //     for (int j = 0; j < 3; j++)
    //         Fncandid_mat(i,j) = Fncandid(i,j);

    // std::cout << "Fncandid (normalized space):" << std::endl; printM_here(Fncandid_mat);

    // STEP 4: we denormalize F to get Fcandidate
    Fcandid=transpose(N2)*Fncandid*N1;

    // // STEP 5: we re-inforce rank 2 of F again after denormalization
    // FMatrix<float,3,3> Uf2;
    // FMatrix<float,3,3> Vf2;
    // FVector<float,3> Sf2;
    // svd(Fcandid, Uf2, Sf2, Vf2);
    // Sf2[2] = 0;

    // // changing the Sf2 vector to a 3x3 matrix:
    // FMatrix<float,3,3> Sfm2; 
    // Sfm2.fill(0.0f);
    // for(int id=0;id<3;id++){
    //     for(int j=0;j<3;j++){
    //         if(id==j){
    //             Sfm2(id,id)=Sf2[id];
    //         }
    //     }
    // }

    // Fcandid=Uf2*Sfm2*Vf2;    

    return Fcandid;      
}

// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<double,3,3> computeF(vector<SiftMatch>& matches) {
    const double distMax = 1.5; // Pixel error for inlier/outlier discrimination
    // 100000
    int Niter=100000; // Adjusted dynamically
    FMatrix<double,3,3> bestF;

    bestF.fill(0.0f); // safe default

    vector<int> bestInliers; // has the indices of the matches considered inliers for bestF.

    // we make sure matches has at least 8 correspondences
    if(matches.size() < 8) {
        cerr << "Not enough matches to estimate F (need >= 8)" << endl;
        return FMatrix<double,3,3>();
    }

    // --------------- TODO ------------
    // DO NOT FORGET NORMALIZATION OF POINTS

    FMatrix<double,3,3> N1 ;
    FMatrix<double,3,3> N2;

    vector<FMatrix<double,3,3>> N_list;
    N_list=compute_N(matches);
    N1=N_list[0];
    N2=N_list[1];

    std::mt19937 gen(42); // fixed seed — reproducible across runs only for debugging! 

    // uncomment the below and comment the seed for prduction
    // std::random_device rd;
    // std::mt19937 gen(rd());

    for(int i=0; i<Niter; i++)
    {
        // STEP 1: we randomly pick 8 correspondences of the matches. k=8 

        vector<SiftMatch> subset;
        int nMatches = matches.size();

        std::uniform_int_distribution<> dist(0, nMatches - 1);

        vector<int> chosen; // to avoid duplicates, chosen keeps track of the indices already selected to be part of the subset
        while (subset.size() < 8) {
            int idx = dist(gen);

            // we make sure we don't pick the same match twice
            // searching chosen vector for the value of index (idx), it returns chosen.end() if the index is not found
            if (find(chosen.begin(), chosen.end(), idx)==chosen.end()) {
                subset.push_back(matches[idx]);
                chosen.push_back(idx);
            }
        }

        // STEP 2: we apply the 8-point algorithm on the subset chosen to get Fcandidate 

        FMatrix<double,3,3> Fcandid=eightpointalgo(subset, N1, N2);
        
        // if the Fcandid returned is all 0s that means the submatches of this RANSAC iteration gave a degenerate A matrix, we skip this iteration of RANSAC
        bool isZero = true;
        for(int a=0;a<3 && isZero;++a){
            for(int b=0;b<3;++b){
                if(fabs(Fcandid(a,b))>1e-12f){
                    isZero=false; 
                    break; 
                }
            } 
        } 

        if (isZero) {
            cout << "F is zero in this run!" << endl;
            continue;
        }

        // STEP 3: We count the inliers by using the Sampson error
        
        vector<int> Inliers; // has the indices of the matches considered inliers.
        Inliers=mark_inliers(Fcandid,matches,distMax);

        // STEP 4: we store the indices of the inliers in bestInliers if the number obtained was bigger than the max number obtained so far and store Fcandidate in bestF
        
        int in_nbr=Inliers.size();
        if(in_nbr>bestInliers.size()){

            // here this means we obtained a better F candidate
            bestInliers=Inliers;
            bestF=Fcandid;
            
            // STEP 5: Lastly, if we got a better F, we change Niter dynamically --> Niter=log beta / log(1-(m/n)^k)
            double w = double(in_nbr) / double(matches.size());
            if (w > 0.0f && w < 1.0f) {
                double p = log(BETA) / log(1.0f - std::pow(w, (double)8));
                if (p > 0) Niter = min(Niter, (int)ceilf(p));
            }
        }
    }

    // Updating matches with inliers only
    vector<SiftMatch> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++)
        matches.push_back(all[bestInliers[i]]);

    
    vector<FMatrix<double,3,3>> N_list_final = compute_N(matches);

    // STEP 6: Finally, we re-compute F using all the inliers obtained in bestInliers
    bestF=eightpointalgo(matches, N_list_final[0], N_list_final[1]);
    return bestF;
}

// debugging function

// draws lines between matches: could be hard to visualize if too many matches

// void drawMatches(Window w, Image<Color,2> I1, Image<Color,2> I2, const vector<Match>& matches) {
//     setActiveWindow(w);
//     display(I1, 0, 0);
//     display(I2, I1.width(), 0);
//     for (const auto& m : matches) {
//         drawLine((int)m.x1, (int)m.y1,
//                  (int)m.x2 + I1.width(), (int)m.y2,
//                  Color(255,0,0));
//     }
// }

// draws lines between 80 matches max: easier to see
// void drawMatches(Window w, Image<Color,2> I1, Image<Color,2> I2,
//                   const vector<Match>& matches, int maxToDraw = 80) {
//     setActiveWindow(w);
//     display(I1, 0, 0);
//     display(I2, I1.width(), 0);

//     vector<int> indices(matches.size());
//     for (size_t i = 0; i < indices.size(); i++) indices[i] = i;

//     if ((int)indices.size() > maxToDraw) {
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::shuffle(indices.begin(), indices.end(), gen);
//         indices.resize(maxToDraw);
//     }

//     for (int idx : indices) {
//         const Match& m = matches[idx];
//         drawLine((int)m.x1, (int)m.y1,
//                   (int)m.x2 + I1.width(), (int)m.y2,
//                   Color(255,0,0));
//     }
// }

// draws points for every match: red point in I1, green point in I2.
// void drawMatches(Window w, Image<Color,2> I1, Image<Color,2> I2, const vector<Match>& matches) {
//     setActiveWindow(w);
//     display(I1, 0, 0);
//     display(I2, I1.width(), 0);
//     for (const auto& m : matches) {
//         drawCircle((int)m.x1, (int)m.y1, 3, Color(255,0,0), 2);
//         drawCircle((int)m.x2 + I1.width(), (int)m.y2, 3, Color(0,255,0), 2);
//     }
// }

// Draws matches as colored dots: each match gets a unique color, and the same
// color is used for both the point in image 1 and its corresponding point in image 2.
// No connecting lines - just color-coded dots to visually pair matches without clutter.
void drawMatches(Window w, Image<Color,2> I1, Image<Color,2> I2,
                      const vector<SiftMatch>& matches, int maxToDraw = 150) {
    setActiveWindow(w);
    display(I1, 0, 0);
    display(I2, I1.width(), 0);

    // subsample if there are too many matches to keep it visually readable
    vector<int> indices(matches.size());
    for (size_t i = 0; i < indices.size(); i++) indices[i] = i;

    if ((int)indices.size() > maxToDraw) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);
        indices.resize(maxToDraw);
    }

    // deterministic per-index color generator so re-running with the same
    // subsample order gives the same colors (useful for comparing screenshots)
    std::mt19937 colorGen(12345);
    std::uniform_int_distribution<int> colorDist(50, 255); // avoid near-black colors

    for (int idx : indices) {
        const SiftMatch& m = matches[idx];

        Color c((unsigned char)colorDist(colorGen),
        (unsigned char)colorDist(colorGen),
        (unsigned char)colorDist(colorGen));

        // point in image 1
        fillCircle((int)m.x1, (int)m.y1, 4, c);
        // point in image 2 (shifted by I1's width to sit in the right half)
        fillCircle((int)m.x2 + I1.width(), (int)m.y2, 4, c);
    }
}

vector<SiftMatch> GetInliers(const std::string& I1_path, const std::string& I2_path, Mat& F_RANSAC, double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3){
    vector<SiftMatch> matches;

    // Load images
    Image<Color,2> I1, I2;
    if( ! load(I1, I1_path.c_str()) ||
        ! load(I2, I2_path.c_str()) ) {
        cerr<< "Unable to load images" << endl;
        return matches;
    }

    algoSIFT(I1, I2, matches);
    undistortMatches(matches, fx,  fy,  cx,  cy,  k1,  k2,  p1,  p2,  k3);
    removeDuplicateMatches(matches);
    // std::cout << "raw matches after SIFT+ratio-test+dedup: " << matches.size() << std::endl;

    // debugging: we visualize the matches before RANSAC
    // int W = I1.width() + I2.width();
    // int H = max(I1.height(), I2.height());
    // Window w1 = openWindow(W, H);
    // drawMatches(w1, I1, I2, matches);   // before RANSAC
    // click();
    // closeWindow(w1);

    double sigma = 0.0;
    libNumerics::matrix<double> F = computeF_ORSA(matches, I1.width(), I1.height(), I2.width(), I2.height(), &sigma);

    // FMatrix<double,3,3> F = computeF(matches);
    // cout << "F="<< endl << F;
    const int n = (int)matches.size();
    // std::cout << " matches: " << n << std::endl;

    // debugging: we visualize the matches after RANSAC
    int W2 = I1.width() + I2.width();
    int H2 = max(I1.height(), I2.height());
    Window w2 = openWindow(W2, H2);
    drawMatches(w2, I1, I2, matches);   // after RANSAC
    click();
    closeWindow(w2);

    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            F_RANSAC(i,j)=F(i,j);
        }
    }

    return matches;
}