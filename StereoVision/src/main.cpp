#include <opencv2/opencv.hpp>
#include <iostream>
#include <string> 
#include <fstream>
#include <sstream>
#include "main.h"
#include <omp.h>

int main(int argc, char** argv) {

    ////////////////
    // Parameters //
    ////////////////

    // camera setup parameters
    const double baseline = 213;
    const double focal_length = 1247;

    // stereo estimation parameters
    const int dmin = 128;
    const int window_size = (argc > 4) ? std::stoi(argv[4]) : 5;
    const double weight = (argc > 5) ? std::stod(argv[5]) : 2000;
    const double scale = 3;
    const double cx_d = 2928.3; // Cx and Cy are principal distances of the camera and are its intrinsic properties
    const double cy_d = 940.545;
   // const double offset = 553.54;
    ///////////////////////////
    // Commandline arguments //
    ///////////////////////////

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " IMAGE1 IMAGE2 OUTPUT_FILE, WINDOW_SIZE, WEIGHT" << std::endl;
        return 1;
    }

    cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    const std::string output_file = argv[3];
    //if (argv[4]) window_size = atoi(argv[4]);
    //if (argv[5]) weight = atoi(argv[5]);

    if (!image1.data) {
        std::cerr << "No image1 data" << std::endl;
        return EXIT_FAILURE;
    }

    if (!image2.data) {
        std::cerr << "No image2 data" << std::endl;
        return EXIT_FAILURE;
    }

  
    std::cout << "------------------ Parameters -------------------" << std::endl;
    std::cout << "focal_length = " << focal_length << std::endl;
    std::cout << "baseline = " << baseline << std::endl;
    std::cout << "window_size = " << window_size << std::endl;
    std::cout << "occlusion weights = " << weight << std::endl;
    std::cout << "disparity added due to image cropping = " << dmin << std::endl;
    std::cout << "scaling of disparity images to show = " << scale << std::endl;
    std::cout << "output filename = " << argv[3] << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    int height = image1.size().height;
    int width = image1.size().width;
    ////////////////////
    // Reconstruction //
    ////////////////////

    // Naive disparity image
    cv::Mat naive_disparities = cv::Mat::zeros(height, width, CV_8UC1);
    // DP disparity image
    cv::Mat dp_disparities = cv::Mat::zeros(height, width, CV_8UC1);
   

    StereoEstimation_DynamicProgramming(
        window_size,
        height,
        width,
        weight,
        image1, image2,dp_disparities, scale);

    cv::Mat normalized_dp_disparities;
    cv::normalize(dp_disparities, normalized_dp_disparities, 255, 0, cv::NORM_MINMAX);//normalize to get the full color spectrum

    // save and display images
    std::stringstream out2;
    out2 << output_file << "_dp.png";
    cv::imwrite(out2.str(), normalized_dp_disparities);
  

    cv::namedWindow("Dynamic Programming", cv::WINDOW_AUTOSIZE);
    cv::imshow("Dynamic Programming disparities", normalized_dp_disparities);
    cv::waitKey(0);

    StereoEstimation_Naive(
        window_size, dmin, height, width,
        image1, image2,
        naive_disparities, scale);

    ////////////
    // Output //
    ////////////

    // reconstruction
    Disparity2PointCloud(
        output_file,
        height, width, dp_disparities,
        window_size, dmin, baseline, focal_length, cx_d, cy_d);

    // save and display images
    std::stringstream out1;
    out1 << output_file << "_naive.png";
    cv::imwrite(out1.str(), naive_disparities);

    cv::namedWindow("Naive", cv::WINDOW_AUTOSIZE);
    cv::imshow("Naive", naive_disparities);
    cv::waitKey(0);

    return 0;
}


void StereoEstimation_Naive(
    const int& window_size,
    const int& dmin,
    int height,
    int width,
    cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities, const double& scale)
{
    int half_window_size = window_size / 2;
    //We use half windows we can pad the image and not run out of the image
    for (int i = half_window_size; i < height - half_window_size; ++i) {

        std::cout
            << "Calculating disparities for the naive approach... "
            << std::ceil(((i - half_window_size + 1) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
            << std::flush;
#pragma omp parallel for
        for (int j = half_window_size; j < width - half_window_size; ++j) {//for each pixel on the left image
            int min_ssd = INT_MAX;
            int disparity = 0;
            //We choose the disparity for particular location of the left images
            //that matches best location on right image
            for (int d = -j + half_window_size; d < width - j - half_window_size; ++d) {
                //We go throught entire row of right image
                int ssd = 0;

                // TODO: sum up matching cost (ssd) in a window
                // Consider each location defined by disparity, put a window around it
                // for parametriazing the first coordinate of the image, we from negative to positive window size
                for (int u = -half_window_size; u <= half_window_size; ++u) {
                    for (int v = -half_window_size; v <= half_window_size; ++v)
                    {
                        int val_left = image1.at<uchar>(i + u, j + v);
                        int val_right = image2.at<uchar>(i + u, j + v + d);// disparity
                        //Sum up square differences
                        ssd += (val_left - val_right) * (val_left - val_right);
                    }
                }

                // hold track of the pixel with lowest SSD error
                if (ssd < min_ssd) {
                    min_ssd = ssd;
                    disparity = d;
                }
            }

            naive_disparities.at<uchar>(i - half_window_size, j - half_window_size) = std::abs(disparity) * scale;
        }
    }

    std::cout << "Calculating disparities for the naive approach... Done.\r" << std::flush;
    std::cout << std::endl;
}


void StereoEstimation_DynamicProgramming(
    const int& window_size,
    int height,
    int width,
    int weight,
    cv::Mat& image1, cv::Mat& image2, cv::Mat& dp_disparities,
     const double& scale) {
    
    int disp_x, disp_y = 0;
    int half_window_size = window_size / 2;
    for (int row = half_window_size; row < height - half_window_size; ++row)
    {
        cv::Mat C = cv::Mat::zeros(width, width, CV_32F);//Cost Matrix
        cv::Mat M = cv::Mat::ones(width, width, CV_8UC1);//Disparity space image
        std::cout
            << "Calculating disparities for the Dynamic Programming approach...  "
            << std::ceil(((row - half_window_size + 1) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
            << std::flush;

        C.at<float>(0, 0) = weight;
        M.at<uchar>(0, 0) = 3;

#pragma omp parallel for
        for (int i = half_window_size + 1; i < width - half_window_size; ++i) {
            C.at<float>(i , 0) = C.at<float>(i - 1, 0) + weight; 
            M.at<uchar>(i , 0) = 2; // left occlusion
            C.at<float>(0, i ) += C.at<float>(0, i - 1) + weight;
            M.at<uchar>(0, i ) = 3; // right occlusion
        }
#pragma omp parallel for
        for (int i = half_window_size + 1; i < width - half_window_size; ++i) {//left scanline
            for (int j = half_window_size + 1; j < width - half_window_size; ++j) {//right scanline
                // TODO: sum up matching cost (ssd) in a window
                int dissimilarity = 0;
                for (int u = -half_window_size; u <= half_window_size; ++u) {
                    for (int v = -half_window_size; v <= half_window_size; ++v)
                    {
                        int val_left = image1.at<uchar>(row + u, i + v);
                        int val_right = image2.at<uchar>(row + u, j + v);

                        dissimilarity += (val_left - val_right) * (val_left - val_right);
                    }
                }
                //from source to sink
                int min1 = C.at<float>(i  - 1, j  - 1) + dissimilarity;//match
                int min2 = C.at<float>(i  - 1, j ) + weight; // left occlusion
                int min3 = C.at<float>(i , j  - 1) + weight; // right occlusion
                int min_c = std::min({ min1, min2, min3 });
                C.at<float>(i , j ) = min_c; // Finding optimal cost, update the cost matrix

                if (min_c == min1) M.at<uchar>(i , j ) = 1;//match
                else if (min_c == min2) M.at<uchar>(i , j ) = 2;  // left occlusion
                else if (min_c == min3) M.at<uchar>(i , j ) = 3;  // right occlusion

            }
        }
        // starting from the bottom right, from sink to source 
        disp_x = width - 1;
        disp_y = width - 1;
        int w = width;
        int d = 0;
        while (disp_x != 0 && disp_y != 0) {
            switch (M.at<uchar>(disp_x, disp_y)) {
                case 1:
                    d = abs(disp_x - disp_y);
                    disp_x--;
                    disp_y--;
                    w--;
                    break;
                case 2:
                    disp_x--;
                    break;
                case 3:
                    disp_y--;
                    break;
            }
            dp_disparities.at<uchar>(row - half_window_size, w - half_window_size) = d;
           
        }

    }
}

    void Disparity2PointCloud(
    const std::string& output_file,
    int height, int width, cv::Mat& disparities,
    const int& window_size,
    const int& dmin, const double& baseline, const double& focal_length, const double& cx_d, const double& cy_d)
{
    //We need calibration info
    std::stringstream out3d;
    out3d << output_file << ".xyz";
    std::ofstream outfile(out3d.str());

    for (int i = 0; i < height - window_size; ++i) {
        std::cout << "Reconstructing 3D point cloud from disparities... " << std::ceil(((i) / static_cast<double>(height - window_size + 1)) * 100) << "%\r" << std::flush;
        for (int j = 0; j < width - window_size; ++j) {
            if (disparities.at<uchar>(i, j) == 0) continue;
            // TODO, change focal length if we change sizze of the image
            double Z = (baseline * focal_length) / (disparities.at<uchar>(i, j)+dmin);
            double X = (i - cx_d) * Z / focal_length;// We substract the principal distance of camera cx and cy from X and Y respectively
            double Y = (j - cy_d) * Z / focal_length;
            outfile << X << " " << Y << " " << Z << std::endl;
        }
    }

    std::cout << "Reconstructing 3D point cloud from disparities... Done.\r" << std::flush;
    
}