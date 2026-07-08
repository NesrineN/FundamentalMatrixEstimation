#include "libOrsa/libNumerics/matrix.h"
#include "FNS.h"
#include "GaussNewton.h"
#include "Initialization.h"
#include "PoseEstimation.h"
#include "Pipeline.h"
#include "GetInliers.h"

#include <vector>
#include <numeric>  
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <iostream> 
#include <cmath>
#include <algorithm>

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

using namespace Imagine;
using namespace std;

struct Association
{
    std::string rgb_path;

    double tx, ty, tz;
    double qx, qy, qz, qw;
};

std::vector<Association> loadAssociations(const std::string& dataset_path)
{
    std::vector<Association> data;

    std::string file_path = dataset_path + "/associations.txt";

    std::ifstream file(file_path);

    if (!file.is_open())
    {
        std::cerr << "Failed to open: " << file_path << std::endl;
        return data;
    }

    std::string line;

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        std::stringstream ss(line);

        double timestamp; // ignored
        Association a;

        ss >> timestamp;       // skip
        ss >> a.rgb_path;      // rgb image path

        ss >> a.tx >> a.ty >> a.tz;
        ss >> a.qx >> a.qy >> a.qz >> a.qw;

        data.push_back(a);
    }

    return data;
}

Mat quatToRot(double x, double y, double z, double w)
{
    double qnorm = std::sqrt(x*x + y*y + z*z + w*w);
    std::cout << "quaternion norm: " << qnorm << std::endl; // should be ~1.0

    Mat R=Mat::zeros(3);

    R(0,0) = 1 - 2*(y*y + z*z);
    R(0,1) = 2*(x*y - z*w);
    R(0,2) = 2*(x*z + y*w);

    R(1,0) = 2*(x*y + z*w);
    R(1,1) = 1 - 2*(x*x + z*z);
    R(1,2) = 2*(y*z - x*w);

    R(2,0) = 2*(x*z - y*w);
    R(2,1) = 2*(y*z + x*w);
    R(2,2) = 1 - 2*(x*x + y*y);

    return R;
}

Mat orthogonalizeRotation(const Mat& R) {
    Mat U(3,3), V(3,3);
    Vec S(3);
    R.SVD(U, S, V);
    Mat R_ortho = U * V.t();
    if (R_ortho.det() < 0) {
        for (int i=0;i<3;i++) U(i,2) = -U(i,2);
        R_ortho = U * V.t();
    }
    return R_ortho;
}

// this function takes R1 t1 and R2 t2 and computes relative pose R and t
void computeRelativePose(
    const Mat& R1, const Vec& t1,
    const Mat& R2, const Vec& t2,
    Mat& R_rel, Vec& t_rel)
{
    Mat R1o = orthogonalizeRotation(R1);
    Mat R2o = orthogonalizeRotation(R2);

    R_rel = R2o.t() * R1o;
    R_rel = orthogonalizeRotation(R_rel); // clean up any residual multiplication drift too

    t_rel = R2o.t() * (t1 - t2);
}

double computeMean(const std::vector<double>& v)
{
    if (v.empty()) return 0.0;

    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / static_cast<double>(v.size());
}

double computeMedian(std::vector<double> v) 
{
    if (v.empty()) return 0.0;

    std::sort(v.begin(), v.end());

    size_t n = v.size();
    size_t mid = n / 2;

    if (n % 2 == 0)
    {
        return (v[mid - 1] + v[mid]) * 0.5;
    }
    else
    {
        return v[mid];
    }
}

void exportErrorsCSV(
    const std::vector<double>& rot_fns,
    const std::vector<double>& trans_fns,
    const std::vector<double>& rot_gauss,
    const std::vector<double>& trans_gauss,
    const std::string& filename)
{
    std::ofstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Cannot open file for writing\n";
        return;
    }

    file << "rot_fns,trans_fns,rot_gauss,trans_gauss\n";

    size_t n = std::max({
        rot_fns.size(),
        trans_fns.size(),
        rot_gauss.size(),
        trans_gauss.size()
    });

    for (size_t i = 0; i < n; i++)
    {
        if (i < rot_fns.size()) file << rot_fns[i]; else file << "nan";
        file << ",";

        if (i < trans_fns.size()) file << trans_fns[i]; else file << "nan";
        file << ",";

        if (i < rot_gauss.size()) file << rot_gauss[i]; else file << "nan";
        file << ",";

        if (i < trans_gauss.size()) file << trans_gauss[i]; else file << "nan";

        file << "\n";
    }
}

// ----------------------------------------------------------------------------------------------------------------------------------------------

int main(int argc, char* argv[]){

    // Intrinsics provided by TUM RGB-D Dataset for Freiburg 1 RGB data:
    double fx=517.3, fy=516.5, cx=318.6, cy=255.3;
    // distortion parameters provided by TUM RGB-D
    double k1=0.2624;
    double k2=-0.9531;
    double p1=-0.0054;
    double p2=0.0026;
    double k3=1.1633;

    Mat K=Mat::eye(3);
    K(0,0)=fx;
    K(0,2)=cx;
    K(1,1)=fy;
    K(1,2)=cy;

    double f0 = (fx + fy) * 0.5;
    
    // Pre-processing of the dataset:
    std::ifstream file("config/dataset.txt");
    std::string line;

    std::string dataset_path;

    while (std::getline(file, line))
    {
        if (line.find("dataset_path=") == 0)
        {
            dataset_path = line.substr(std::string("dataset_path=").size());
        }
    }

    // we want to extract the associations from the associations.txt file. 
    // they have the following format:
    // timestamp rgb/path.png tx ty tz qx qy qz qw
    // we store them in a vector of Associations where each Association has rgb path and the corresponding tx ty tz qx qy qz qw
    std::vector<Association> associations = loadAssociations(dataset_path);

    // now, we loop over the associations and run the pipeline to get the rotation and translation errors: 
    std::vector<double> Rotation_errors_FNS;
    std::vector<double> Translation_errors_FNS;

    std::vector<double> Rotation_errors_Gauss;
    std::vector<double> Translation_errors_Gauss;

    for (size_t i = 0; i + 12 < associations.size(); i++)
    {
        const auto& a1 = associations[i];
        const auto& a2 = associations[i + 12];

        Mat R1 = quatToRot(a1.qx, a1.qy, a1.qz, a1.qw);
        Mat R2 = quatToRot(a2.qx, a2.qy, a2.qz, a2.qw);

        Vec t1(3), t2(3);

        t1(0) = a1.tx; t1(1) = a1.ty; t1(2) = a1.tz;
        t2(0) = a2.tx; t2(1) = a2.ty; t2(2) = a2.tz;

        Mat R_rel_gt(3,3);
        Vec t_rel_gt(3);

        computeRelativePose(R1, t1, R2, t2, R_rel_gt, t_rel_gt);

        std::cout << "square magnitude of t_gt is: " << t_rel_gt.qnorm() << std::endl;

        // debugging:
        // std::cout << "R ground truth is: " << std::endl;
        // printM(R_rel_gt);

        // std::cout << "t ground truth is: " << std::endl;
        // printV(t_rel_gt);

        // image paths
        std::string I1_path = a1.rgb_path;
        std::string I2_path = a2.rgb_path;

        I1_path = dataset_path + "\\" + I1_path;
        I2_path = dataset_path + "\\" + I2_path;

        std::cout << "Image 1 path is: " << I1_path << std::endl;
        std::cout << "Image 2 path is: " << I2_path << std::endl;

        // Load images
        Image<Color,2> I1, I2;
        if( ! load(I1, I1_path.c_str()) ||
            ! load(I2, I2_path.c_str()) ) {
            cerr<< "Unable to load images" << endl;
        }

        // now we run the pipeline to get the errors:
        Vec errors_FNS=RunPipelineNoiseless(I1, I2, I1_path, I2_path, K, K, f0, R_rel_gt, t_rel_gt, 1, fx,  fy,  cx,  cy, k1,  k2,  p1,  p2,  k3);
        // Vec errors_Gauss=RunPipelineNoiseless(I1_path, I2_path, K, K, f0, R_rel_gt, t_rel_gt, 2);


        Rotation_errors_FNS.push_back(errors_FNS(0));
        // Rotation_errors_Gauss.push_back(errors_Gauss(0));

        Translation_errors_FNS.push_back(errors_FNS(1));
        // Translation_errors_Gauss.push_back(errors_Gauss(1));
    }

    // we export the errors so we can visualize them later using python:
    // exportErrorsCSV(Rotation_errors_FNS, Translation_errors_FNS, Rotation_errors_Gauss, Translation_errors_Gauss, "pose_errors_classical.csv");

    // now we print the mean/median of the errors:
    double mean_rotation_FNS  = computeMean(Rotation_errors_FNS);
    double median_rotation_FNS = computeMedian(Rotation_errors_FNS);

    double mean_translation_FNS  = computeMean(Translation_errors_FNS);
    double median_translation_FNS = computeMedian(Translation_errors_FNS);

    // double mean_rotation_Gauss  = computeMean(Rotation_errors_Gauss);
    // double median_rotation_Gauss = computeMedian(Rotation_errors_Gauss);

    // double mean_translation_Gauss  = computeMean(Translation_errors_Gauss);
    // double median_translation_Gauss = computeMedian(Translation_errors_Gauss);


    std::cout << "Mean rotation error - FNS: " << mean_rotation_FNS << std::endl;
    std::cout << "Median rotation error - FNS: " << median_rotation_FNS << std::endl;

    // std::cout << "Mean rotation error - Gauss: " << mean_rotation_Gauss << std::endl;
    // std::cout << "Median rotation error - Gauss: " << median_rotation_Gauss << std::endl;

    std::cout << "Mean translation error - FNS: " << mean_translation_FNS << std::endl;
    std::cout << "Median translation error - FNS: " << median_translation_FNS << std::endl;

    // std::cout << "Mean translation error - Gauss: " << mean_translation_Gauss << std::endl;
    // std::cout << "Median translation error - Gauss: " << median_translation_Gauss << std::endl;

    return 0;

}