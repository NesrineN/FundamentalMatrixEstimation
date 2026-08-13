// Builds the homography induced by a plane with normal n (in camera 1 frame)
// at distance d from camera 1's center, given relative pose (R, t) to camera 2.
// H maps points in image 1 to their corresponding points in image 2 (unwarped).
//
// Standard formula: H = K * (R - (t * n^T) / d) * K^-1
Mat buildHomography(const Mat& K, const Mat& R, const Vec& t, const Vec& n, double d) {
    Mat tn = Mat::zeros(3,3);
    for (int i=0;i<3;i++)
        for (int j=0;j<3;j++)
            tn(i,j) = t(i) * n(j);

    Mat Rtn = R - tn / d;
    Mat Kinv = K.inv();

    return K * Rtn * Kinv;
}

// Warps I1 by homography H to produce I2 (same size as I1).
// Uses inverse warping: for each pixel in the OUTPUT image, find its
// corresponding source location in I1 via H^-1, and bilinearly interpolate.
Image<Color,2> warpImageByHomography(const Image<Color,2>& I1, const Mat& H, int outW, int outH) {
    Image<Color,2> I2(outW, outH);
    Mat Hinv = H.inv();

    for (int v = 0; v < outH; v++) {
        for (int u = 0; u < outW; u++) {
            Vec p(3);
            p(0) = u; p(1) = v; p(2) = 1.0;
            Vec src = Hinv * p;
            double sx = src(0) / src(2);
            double sy = src(1) / src(2);

            if (sx < 0 || sx >= I1.width()-1 || sy < 0 || sy >= I1.height()-1) {
                I2(u,v) = Color(0,0,0); // out of bounds -> black
                continue;
            }

            int x0 = (int)sx, y0 = (int)sy;
            double fx = sx - x0, fy = sy - y0;

            Color c00 = I1(x0,   y0);
            Color c10 = I1(x0+1, y0);
            Color c01 = I1(x0,   y0+1);
            Color c11 = I1(x0+1, y0+1);

            auto lerp = [](double a, double b, double t){ return a + t*(b-a); };

            for (int ch = 0; ch < 3; ch++) {
                double top = lerp(c00[ch], c10[ch], fx);
                double bot = lerp(c01[ch], c11[ch], fx);
                double val = lerp(top, bot, fy);
                I2(u,v)[ch] = (unsigned char)std::round(std::max(0.0, std::min(255.0, val)));
            }
        }
    }
    return I2;
}

void runHomographyWarpTest(const std::string& imagePath) {
    std::cout << "===== HOMOGRAPHY-WARPED IMAGE PIPELINE TEST =====" << std::endl;

    // 1. Load a real image
    Image<Color,2> I1;
    if (!load(I1, imagePath.c_str())) {
        std::cerr << "Unable to load image: " << imagePath << std::endl;
        return;
    }

    // 2. Camera intrinsics (use your TUM fr1 values, or synthetic ones)
    double fx=517.3, fy=516.5, cx=318.6, cy=255.3;
    Mat K = Mat::eye(3);
    K(0,0)=fx; K(1,1)=fy; K(0,2)=cx; K(1,2)=cy;
    Mat K2 = K;

    // 3. Known ground-truth relative pose (moderate rotation, avoids small-rotation ambiguity)
    Mat R_true = makeTestRotation(0.3, 0.15, 0.1); // ~20 deg combined
    Vec t_true(3);
    t_true(0) = 0.1; t_true(1) = 0.02; t_true(2) = -0.05;

    // 4. Fictitious plane: normal roughly facing the camera, at some reasonable depth
    Vec n(3);
    n(0) = 0.0; n(1) = 0.0; n(2) = 1.0; // fronto-parallel plane
    double d = 3.0; // plane distance (meters, arbitrary consistent scale)

    // 5. Build homography and warp I1 -> I2
    Mat H = buildHomography(K, R_true, t_true, n, d);
    Image<Color,2> I2 = warpImageByHomography(I1, H, I1.width(), I1.height());

    // (optional) save I2 to disk to visually inspect the warp
    // save(I2, "warped_image2.png");

    // 6. Run the SAME matching + estimation pipeline used on real TUM pairs
    std::vector<Match> matches;
    algoSIFT(I1, I2, matches);
    removeDuplicateMatches(matches);
    std::cout << "Raw matches after SIFT + dedup: " << matches.size() << std::endl;

    double sigma_orsa = 0.0;
    Mat F_RANSAC = computeF_ORSA(matches, I1.width(), I1.height(), I2.width(), I2.height(), &sigma_orsa);
    std::cout << "Inliers kept: " << matches.size() << std::endl;
    std::cout << "ORSA sigma: " << sigma_orsa << std::endl;

    Mat F_RANSAC_for_FNS = F_RANSAC; // no transpose, per your established ORSA convention

    std::vector<Point2D> img1Pts, img2Pts;
    for (const auto& m : matches) {
        Point2D p1, p2;
        p1.x = m.x1; p1.y = m.y1;
        p2.x = m.x2; p2.y = m.y2;
        img1Pts.push_back(p1);
        img2Pts.push_back(p2);
    }

    double f0 = 600.0;
    Mat F_estimated = GetF(img1Pts, img2Pts, F_RANSAC_for_FNS, f0, /*method=*/1);

    Mat P2 = EstimatePose(I1, I2, K, K2, F_estimated, img1Pts, img2Pts, R_true, t_true);

    Mat R_pred = P2.copy(0,2,0,2);
    Mat t_pred_mat = P2.copyCols(3,3);
    Vec t_pred(3);
    t_pred(0)=t_pred_mat(0,0); t_pred(1)=t_pred_mat(1,0); t_pred(2)=t_pred_mat(2,0);

    double rot_err = rotation_error(R_true, R_pred);
    double trans_err = translation_error(t_true, t_pred);

    std::cout << "Rotation error: " << rot_err << " deg" << std::endl;
    std::cout << "Translation error: " << trans_err << " deg" << std::endl;
    std::cout << "===== END TEST =====" << std::endl;
}

