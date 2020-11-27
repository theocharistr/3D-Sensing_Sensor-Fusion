#pragma once
void StereoEstimation_DynamicProgramming(
    const int& window_size,
    int height,
    int width,
    int weight,
    cv::Mat& image1, cv::Mat& image2, cv::Mat& dp_disparities, const double& scale);

void StereoEstimation_Naive(
    const int& window_size,
    const int& dmin,
    int height,
    int width,
    cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities, const double& scale);

void Disparity2PointCloud(
    const std::string& output_file,
    int height, int width, cv::Mat& disparities,
    const int& window_size,
    const int& dmin, const double& baseline, const double& focal_length, const double& cx_d, const double& cy_d);
