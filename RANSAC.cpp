// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse
// Student: Nesrine Naaman

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>

#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};


// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}

// Function for computing the Normalization Matrices that we will use to normalize the matches
// Isotropic scaling taken from https://www.r-5.org/files/books/computers/algo-list/image-processing/vision/Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision-EN.pdf (P.107)
// "The points are translated so that their centroid is at the origin".
// "The points are then scaled so that the average distance from the origin is equal to √2".
// "This transformation is applied to each of the two images independently".
vector<FMatrix<float,3,3>> compute_N(vector<Match>& matches){
    
    vector<FMatrix<float,3,3>> N_list;
    FMatrix<float,3,3> N1; // the Normalization matrix for Image 1
    FMatrix<float,3,3> N2; // the Normalization matrix for Image 2

    // we will calculate the centroid of points of each image, then the distance from all points of an image to its centroid, get the scale and fill the matrix with it and the centroid
    // the aim is to translate the points so their centroid is at the origin and to scale the points so that the average distance from the origin is sqrt(2). 

    // for I1:
    float xbar1=0.f;
    float ybar1=0.f;
    for(int j=0;j<matches.size();j++){
        xbar1=xbar1+matches[j].x1;
        ybar1=ybar1+matches[j].y1;
    }
    xbar1/=(float)matches.size(); // centroid
    ybar1/=(float)matches.size(); // centroid

    float distance1=0.f;

    for(int ii=0;ii<matches.size();ii++){
        distance1=distance1+(sqrt(pow((matches[ii].x1-xbar1),2)+pow((matches[ii].y1-ybar1),2)));
    }

    distance1/=(float)matches.size(); // distance of points from the centroid
    if(distance1 < 1e-8f) distance1 = 1.0f; // to avoid dividing by zero in case distance is very close to zero
    float s1=sqrt(2.0f)/distance1; // scale to normalize by
    
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
    float xbar2=0.f;
    float ybar2=0.f;
    for(int id=0;id<matches.size();id++){
        xbar2=xbar2+matches[id].x2;
        ybar2=ybar2+matches[id].y2;
    }
    xbar2/=(float)matches.size();
    ybar2/=(float)matches.size();
    float distance2=0.f;
    for(int ind=0;ind<matches.size();ind++){
        distance2=distance2+(sqrt(pow((matches[ind].x2-xbar2),2)+pow((matches[ind].y2-ybar2),2)));
    }
    distance2/=(float)matches.size();
    if(distance2 < 1e-8f) distance2 = 1.0f;    
    float s2=sqrt(2.0f)/distance2;

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
vector<Match> normalize_matches(FMatrix<float,3,3> N1, FMatrix<float,3,3> N2, vector<Match>& matches){
    
    vector<Match> subset_normalized;
    for(int match_id=0;match_id<matches.size(); match_id++){
        Match m=matches[match_id];
        Match m_normalized;

        FVector<float,3> X1h, X2h; // point correspondences in a match in homogeneous system
        X1h[0]= m.x1;  X1h[1]= m.y1;  X1h[2]= 1.0f;
        X2h[0]= m.x2;  X2h[1]= m.y2;  X2h[2]= 1.0f;

        // we normalize both
        FVector<float,3> X1n= N1 * X1h;
        FVector<float,3> X2n= N2 * X2h;

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
vector<int> mark_inliers(FMatrix<float,3,3>& Fcandid, vector<Match>& matches, float distMax){
        
        vector<int> Inliers; // has the indices of the matches considered inliers.
        for(int id=0; id<matches.size();id++){
            Match m=matches[id];

            FVector<float,3> X1h, X2h; // point correspondences in a match in homogeneous system
            X1h[0]= m.x1;  X1h[1]= m.y1;  X1h[2]= 1.0f;
            X2h[0]= m.x2;  X2h[1]= m.y2;  X2h[2]= 1.0f;

            float l[3]; // this is the epipolar line F.xi
            for (int j = 0; j < 3; j++){
                l[j] = Fcandid(j,0)*X1h[0] + Fcandid(j,1)*X1h[1] + Fcandid(j,2)*X1h[2];
            }

            float l2[3]; // this is the epipolar line F^T.x'i
            FMatrix<float,3,3> Ftcandid = transpose(Fcandid);
            for (int j = 0; j < 3; j++){
                l2[j] = Ftcandid(j,0)*X2h[0] + Ftcandid(j,1)*X2h[1] + Ftcandid(j,2)*X2h[2];
            }            

            // now we calculate the sampson distance
            
            // we split the calculation of the epipolar constraint for better ease
            // Step 1: Fx = F * x
            float Fx[3];
            for (int i = 0; i < 3; ++i) {
                Fx[i] = 0.0f;
                for (int j = 0; j < 3; ++j) {
                    Fx[i] += Fcandid(i, j) * X1h[j];  
                }
            }

            // Step 2: s = x'^T * Fx
            float s = 0.0f;
            for (int i = 0; i < 3; ++i) {
                s += X2h[i] * Fx[i];
            }

            // Step 3: numerator
            float numerator = s * s;


            float denom=l[0]*l[0] + l[1]*l[1] + l2[0]*l2[0] + l2[1]*l2[1];

            if(denom<1e-8f){denom=1e-8f;} // to make sure we don't divide by zero
            
            float di2 = numerator/denom; 
            float di=sqrt(di2);
            
            if(di<=distMax){
                // m is an inlier, we add it to the temporary vector Inliers which stores the inliers' indices 
                Inliers.push_back((int)id);
            }
        }
        return Inliers;
}

// Function that calculates the Fundamental Matrix using correspondences in matches vector. It returns the F computed and the rank of the matrix A (done to skip samples in RANSAC that produced a degenerate A)
FMatrix<float,3,3> eightpointalgo(vector<Match>& matches){
    
    FMatrix<float,3,3> Fcandid; // to be returned 

    int n=matches.size(); // number of matches we have in our case
    // we need AT LEAST 8 points
    if(n < 8){
    Fcandid.fill(0.0f);
    return Fcandid;
    }

    // STEP 1: We normalize the matches

    FMatrix<float,3,3> N1 ;
    FMatrix<float,3,3> N2;

    vector<FMatrix<float,3,3>> N_list;
    N_list=compute_N(matches);
    N1=N_list[0];
    N2=N_list[1];

    vector<Match> subset_normalized;
    subset_normalized=normalize_matches(N1, N2, matches);

    // STEP 2: We create matrix A and get its SVD

    // A was created using the epipolar constraint x'^T.F.x=0 

    Matrix<float> A(n,9);

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
    Matrix<float> U(n,n);
    Matrix<float> V(9,9);
    int minimum=min(n,9);
    Vector<float>S(minimum);
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

    Vector<float>f(9);
    f=V.getRow(V.nrow() - 1); // this is the column V9 which is the solution to Af=0 (the svd function returns V as V^T that's why we took row instead of column)

    // Reshaping the vector f into a matrix Fncandid (Fnormalized-candidate)
    FMatrix<float,3,3> Fncandid;
    for (int id=0; id<3; id++){
        for (int j=0; j<3; j++){
            Fncandid(id,j)= f[id*3 + j];
        }
    }

    // STEP 3: we set singular value in 3rd column to 0 and recompute Fnormalized-candidate
    FMatrix<float,3,3> Uf;
    FMatrix<float,3,3> Vf;
    FVector<float,3> Sf;
    svd(Fncandid, Uf, Sf, Vf);
    Sf[2] = 0;

    // changing the Sf vector to a 3x3 matrix:
    FMatrix<float,3,3> Sfm; 
    Sfm.fill(0.0f);
    for(int id=0;id<3;id++){
        for(int j=0;j<3;j++){
            if(id==j){
                Sfm(id,id)=Sf[id];
            }
        }
    }

    Fncandid=Uf*Sfm*Vf;

    // STEP 4: we denormalize F to get Fcandidate
    Fcandid=transpose(N2)*Fncandid*N1;

    // STEP 5: we re-inforce rank 2 of F again after denormalization
    FMatrix<float,3,3> Uf2;
    FMatrix<float,3,3> Vf2;
    FVector<float,3> Sf2;
    svd(Fcandid, Uf2, Sf2, Vf2);
    Sf2[2] = 0;

    // changing the Sf2 vector to a 3x3 matrix:
    FMatrix<float,3,3> Sfm2; 
    Sfm2.fill(0.0f);
    for(int id=0;id<3;id++){
        for(int j=0;j<3;j++){
            if(id==j){
                Sfm2(id,id)=Sf2[id];
            }
        }
    }

    Fcandid=Uf2*Sfm2*Vf2;    
    return Fcandid;      
}

// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches) {
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    // 100000
    int Niter=100000; // Adjusted dynamically
    FMatrix<float,3,3> bestF;
    vector<int> bestInliers; // has the indices of the matches considered inliers for bestF.
    
    // --------------- TODO ------------
    // DO NOT FORGET NORMALIZATION OF POINTS

    // we make sure matches has at least 8 correspondences
    if(matches.size() < 8) {
        cerr << "Not enough matches to estimate F (need >= 8)" << endl;
        return FMatrix<float,3,3>();
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    for(int i=0; i<Niter; i++)
    {
        // STEP 1: we randomly pick 8 correspondences of the matches. k=8 

        vector<Match> subset;
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

        FMatrix<float,3,3> Fcandid=eightpointalgo(subset);
        
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
            float w = float(in_nbr) / float(matches.size());
            if (w > 0.0f && w < 1.0f) {
                float p = log(BETA) / log(1.0f - powf(w, (double)8));
                if (p > 0) Niter = min(Niter, (int)ceilf(p));
            }
        }
    }

    // Updating matches with inliers only
    vector<Match> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++)
        matches.push_back(all[bestInliers[i]]);

    // STEP 6: Finally, we re-compute F using all the inliers obtained in bestInliers
    bestF=eightpointalgo(matches);
    return bestF;
}


// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float,3,3>& F) {
    while(true) {
        int x,y; // coordinates of point clicked
        if(getMouse(x,y) == 3)
            break;
        // --------------- TODO ------------
        // if point clicked in I1 --> point x and matching point is x' which lies on the line F.x
        // if point clicked in I2 --> point x' and matching point is x which lies on the line F^Tx'

        if(x>=0 && x<I1.width() && y>=0 && y<I1.height()){
            cout << "You clicked a point in I1" << endl;
            // user clicked point in I1
            FVector<float,3> l; // this is the epipolar line F.x
            FVector<float,3> Xh; // this is the point clicked x in homogeneous coordinates 
            Xh[0]=x;
            Xh[1]=y;
            Xh[2]=1.0f;

            l=F*Xh; // the vector l has a, b, c such that the line equation is ax'+by'+c=0 which is the equation of the epipolar line in I2 where the point X' lies

            // now we draw the line obtained in I2: we take two points:
            // x0=0 (left edge), y0=solution of equation of line
            // x1=I2.width() - 1 (right edge), y1=solution of equation of line

            // if b=0 --> vertical line: we take two points:
            // y0=0 (top edge), x0=solution of equation of line
            // y1=I2.height() - 1 (bottom edge), x1=solution of equation of line

            float x0,x1,y0,y1;
            
            if(fabs(l[1]) < 1e-8f){ // this means it's a vertical line . not enough: because some lines would be deemed vertical when irl they're not. relative !!! ( should be abs b << sqrt a^2 + b^2) 
                y0=0;
                y1=I2.height()-1;
                x0=x1=-l[2]/l[0];
            }
            else{
                cout << "we are here" << endl;
                x0=0;
                x1=I2.width()-1;
                y0=(-(l[0]*x0)-l[2])/(l[1]);
                y1=(-(l[0]*x1)-l[2])/(l[1]);
            }

            // I2 starts at width of I1 and ends at width of I1 + width of I2 so we take the offset into account when using drawLine() function
            drawLine((int)round(x0+I1.width()), (int)round(y0), (int)round(x1+I1.width()), (int)round(y1), RED);  
  
        }

        else if(x>=I1.width() && x<I1.width()+I2.width() && y>=0 && y<I2.height()){
            cout << "You clicked a point in I2" << endl;
            // user clicked point in I2
            FVector<float,3> l; // this is the epipolar line F^Tx'
            FVector<float,3> Xprimeh;
            Xprimeh[0]=x-I1.width(); // removing offset to go to coordinate system of I2
            Xprimeh[1]=y;
            Xprimeh[2]=1.0f;

            l=transpose(F)*Xprimeh; // the vector l has a, b, c such that the line equation is ax+by+c=0 which is the equation of the epipolar line in I1 where the point X lies

            // now we draw the line obtained in I1: we take two points:
            // x0=0 (left edge), y0=solution of equation of line
            // x1=I1.width() - 1 (right edge), y1=solution of equation of line

            // if b=0 --> vertical line: we take two points:
            // y0=0 (top edge), x0=solution of equation of line
            // y1=I1.height() - 1 (bottom edge), x1=solution of equation of line

            int x0,x1,y0,y1;
            
            if(fabs(l[1]) < 1e-6){
                y0=0;
                y1=I1.height()-1;
                x0=x1=-l[2]/l[0];
            }
            else{
                x0=0;
                x1=I1.width()-1;
                y0=(-(l[0]*x0)-l[2])/(l[1]);
                y1=(-(l[0]*x1)-l[2])/(l[1]);


            }
       
            drawLine((int)round(x0), (int)round(y0), (int)round(x1), (int)round(y1), RED);            
        }
        else{
            cout << "You did not click on neither Image 1 nor Image 2" << endl;
        }
    }
}