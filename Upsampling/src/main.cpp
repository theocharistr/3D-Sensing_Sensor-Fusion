#define _USE_MATH_DEFINES
#include <iostream>
#include <opencv2/opencv.hpp>

cv::Mat CreateGaussianKernel(int window_size, float spatial_sigma = 1) {
    cv::Mat kernel(cv::Size(window_size, window_size), CV_32FC1);

    int half_window_size = window_size / 2;

    // see: lecture_03_slides.pdf, Slide 13
    const double k = 2.5;
    const double r_max = std::sqrt(2.0 * half_window_size * half_window_size);
    //const double sigma = r_max / k;

    // sum is for normalization 
    float sum = 0.0;

    for (int x = -window_size / 2; x <= window_size / 2; x++) {
        for (int y = -window_size / 2; y <= window_size / 2; y++) {
            float val = exp(-(x * x + y * y) / (2 * spatial_sigma * spatial_sigma));
            kernel.at<float>(x + window_size / 2, y + window_size / 2) = val;
            sum += val;
        }
    }

    // normalising the Kernel 
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            kernel.at<float>(i, j) /= sum;

    // note that this is a naive implementation
    // there are alternative (better) ways
    // e.g. 
    // - perform analytic normalisation (what's the integral of the gaussian? :))
    // - you could store and compute values as uchar directly in stead of float
    // - computing it as a separable kernel [ exp(x + y) = exp(x) * exp(y) ] ...
    // - ...

    return kernel;
}
cv::Mat OurFilter_Bilateral(const cv::Mat& input, const int window_size = 5, float spatial_sigma = 1, float spectral_sigma = 1) {
    const auto width = input.cols;
    const auto height = input.rows;
    cv::Mat output(input.size(), input.type());

    cv::Mat gaussianKernel = CreateGaussianKernel(window_size, spatial_sigma);

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            output.at<uchar>(r, c) = 0;
        }
    }

    auto d = [](float a, float b) {
        return std::abs(a - b);
    };

    auto p = [](float val, float spectral_sigma) {
        const float sigmaSq = spectral_sigma * spectral_sigma;
        const float normalization = std::sqrt(2 * M_PI) * spectral_sigma;
        return (1 / normalization) * std::exp(-val / (2 * sigmaSq));
    };
    for (int r = window_size / 2; r < height - window_size / 2; ++r) {
        for (int c = window_size / 2; c < width - window_size / 2; ++c) {

            float sum_w = 0;
            float sum = 0;
            //Scan the image with a sliding window, rc is the center of the window
            for (int i = -window_size / 2; i <= window_size / 2; ++i) {
                for (int j = -window_size / 2; j <= window_size / 2; ++j) {
                    //difference in gray level between the center of the window and the current level pixel
                    float range_difference//using the rgb image with the spectral filter
                        = d(input.at<uchar>(r, c), input.at<uchar>(r + i, c + j));
                    //if pixels are different in color -> p is small ,pixel from 2 different areas are less relevant
                    // 2d gaussian -less weight for faraway pixels
                    float w
                        = p(range_difference, spectral_sigma)//sigma for the spectral filter
                        * gaussianKernel.at<float>(i + window_size / 2, j + window_size / 2);

                    sum
                        += input.at<uchar>(r + i, c + j) * w;//using the depth image with the spatial filter
                    //used for normalization
                    sum_w
                        += w;
                }
            }

            output.at<uchar>(r, c) = sum / sum_w;

        }
    }
    return output;
}

void Guided_Joint_Bilateral(const cv::Mat& guidanceIMG, const cv::Mat& depthIMG, cv::Mat& output, const int window_size = 5, float spatial_sigma = 5, float spectral_sigma = 5) {
    const auto width = guidanceIMG.cols;
    const auto height = guidanceIMG.rows;

    cv::Mat gaussianKernel = CreateGaussianKernel(window_size, spatial_sigma);
    auto d = [](float a, float b) {
        return std::abs(a - b);
    };

    auto p = [](float val, float sigma) {
        const float sigmaSq = sigma * sigma;
        const float normalization = std::sqrt(2 * M_PI) * sigma;
        return (1 / normalization) * std::exp(-val / (2 * sigmaSq));
    };

    for (int r = window_size / 2; r < height - window_size / 2; ++r) {
        for (int c = window_size / 2; c < width - window_size / 2; ++c) {

            float sum_w = 0;
            float sum = 0;

            for (int i = -window_size / 2; i <= window_size / 2; ++i) {
                for (int j = -window_size / 2; j <= window_size / 2; ++j) {
                    //we use the guidance image here
                    float range_difference
                        = d(guidanceIMG.at<uchar>(r, c), guidanceIMG.at<uchar>(r + i, c + j));

                    float w
                        = p(range_difference, spectral_sigma)
                        * gaussianKernel.at<float>(i + window_size / 2, j + window_size / 2);

                    sum
                        += depthIMG.at<uchar>(r + i, c + j) * w;
                    sum_w
                        += w;
                }
            }

            output.at<uchar>(r, c) = sum / sum_w;

        }
    }
}

cv::Mat Upsampling(const cv::Mat& guidanceIMG, const cv::Mat& depthIMG) {
    // applying the joint bilateral filter to upsample a depth image, guided by an RGB image  

    int uf = log2(guidanceIMG.rows / depthIMG.rows); // upsample factor
    cv::Mat D = depthIMG; // lowres depth image
    cv::Mat G = guidanceIMG; // highres rgb image
    cv::Mat G_temp;
     
    std:: cout << D.size();
    std::cout << uf;//upsamble factor: how many time i need to double the size of the depth image to have the same resolution of the gray level image
    float downsampleFactor = (1 / pow(4, uf));
    float newSize = downsampleFactor;
    cv::resize(D, D, cv::Size(), downsampleFactor, downsampleFactor, cv::INTER_NEAREST); 
    for (int i = 0; i < uf; ++i)
    {
        // cv::resize(D, D, D.size() * 2);  //double the size of the depth image
        std::cout << D.size();
        cv::resize(G, G_temp, D.size());// resizing the rgb image rgb to depth image
        //applying jont bilateral filter use the gray level image as guidance
        Guided_Joint_Bilateral(G_temp, D, D, 5, 1);
    }
    cv::resize(D, D, guidanceIMG.size());  //resize depth to rgb image
    Guided_Joint_Bilateral(guidanceIMG, D, D, 5, 1);  //apply join bilateral to ful resolution image
    return D;
}
cv::Mat upsample(const cv::Mat& guideImage, const cv::Mat& inputImage) {
    int upsampleFactor = log2(guideImage.rows / inputImage.rows);
    cv::Mat upsampledImage = inputImage;

    for (int i = 1; i < upsampleFactor; ++i) {
        // Doubling the size of the image
        resize(upsampledImage, upsampledImage, cv::Size(), 2, 2);

        // Downscaling guide image
        cv::Mat downscaledGuideImage;
        cv::resize(guideImage, downscaledGuideImage, upsampledImage.size());

        // Filtering upscaled image
        Guided_Joint_Bilateral(upsampledImage, downscaledGuideImage, upsampledImage ,5 ,1);
    }
    resize(upsampledImage, upsampledImage, guideImage.size());
    Guided_Joint_Bilateral(upsampledImage, guideImage, upsampledImage ,5, 1);

    return upsampledImage;
}
int main() {
    cv::Mat img = cv::imread("C:/Users/haris/source/repos/Upsampling/data/view0.png", 0);//grayscale image
    cv::Mat imgRGB = cv::imread("C:/Users/haris/source/repos/Upsampling/data/view0.png");//low resolution depth image
    //RGB image

    cv::imshow("RGB image", imgRGB);
    cv::waitKey();
    cv::Mat input = imgRGB.clone();
    cv::Mat noise(imgRGB.size(), imgRGB.type());
    uchar mean = 0;
    uchar stddev = 25;
    cv::randn(noise, mean, stddev);
    imgRGB += noise;

    cv::Mat UpsampledIMG;

    cv::Mat depthIMG = cv::imread("C:/Users/haris/source/repos/Upsampling/data/disp1.png", 0);//low resolution depth image
    cv::imshow("Low Resolution Depth Image", depthIMG);
    cv::waitKey();

    UpsampledIMG = upsample(imgRGB, depthIMG);
    cv::imshow("Upsampled image", UpsampledIMG);
    cv::waitKey();
    imwrite("C:/Users/haris/source/repos/Upsampling/data/UpsampledIMG.png", UpsampledIMG);

    //spectral filter, also called range kernel
    /*As the range parameter σr increases, the bilateral filter gradually approximates GaussiUpsampledIMGon mpre closely because
    the range Gaussian Gσr widens and flattens, i.e., is nearly
    constant over the intensity interval of the image.
    Increasing the spatial parameter σs smooths larger features.*/
    std::vector < float> spatial_sigmas = { 4,8,16,20 };
    std::vector < float> spectral_sigmas = { 1,5,15,30 };
    std::string r;
    for (float x : spatial_sigmas) {
        for (float y : spectral_sigmas) {
            cv::Mat output = OurFilter_Bilateral(img, 5, x, y);
            std::cout << "x=" << x << ", y=" << y << "\n";
            cv::imshow("OurFilter_Bilateral", output);
            r = "C:/Users/haris/source/repos/Upsampling/data/Bilateral/" + std::to_string(x) + " spatial_sigma, " + std::to_string(y) + " spectral_sigma.png";
            imwrite("C:/Users/haris/source/repos/Upsampling/data/Bilateral/" + std::to_string(x) + " spatial_sigma, " + std::to_string(y) + " spectral_sigma.png", output);
            std::cout << r;

        }
    }


    return 0;

}
