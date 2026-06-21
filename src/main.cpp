#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "cvutils.hpp"
#include "clutils.hpp"

#define TIME_IT(label, expr) \
		[&]() { \
			double _t = cv::getTickCount(); \
			auto _r = (expr); \
			std::cout << (label) << ": " \
			          << (cv::getTickCount() - _t) / cv::getTickFrequency() * 1000.0 \
			          << " ms\n"; \
			return _r; \
		}()

template <typename Func>
double measure_performance(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
	// check for device support
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (auto& platform : platforms) {
        std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (auto& device : devices) {
            std::cout << "  Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        }
    }

    // silence irritating OpenCV warnings
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

	cv::Mat image = cv::imread("../Assets/duck.jpg", cv::IMREAD_COLOR);
	if (image.empty()) {
		std::cerr << "Failed to load image\n";
		return 1;
	}

	// TIME_IT(label, expr) – megmeri expr futasi idejet es kiirja ms-ben

	const int radius = 3; // 7x7-es ablak

	cv::Mat noisyImage   = TIME_IT("Salt & Pepper", addSaltAndPepperNoise(image, 0.1));

	std::cout << "\n--- Filter idok (radius=" << radius << ") ---\n";
	cv::Mat medianResult = TIME_IT("Median CPU", medianFilterCPU(noisyImage, radius));

	cv::imshow("Original",                  image);
	cv::imshow("Noisy (10% salt & pepper)", noisyImage);
	cv::imshow("Median filter (radius=3)",  medianResult);
	image = cv::imread("../assets/duck.jpg", cv::IMREAD_COLOR);

    if(image.empty()) {
        std::cerr << "Could not read the image" << std::endl;
        return 1;
	}

	cv::resize(image, image, cv::Size(512, 512*image.rows/image.cols));
	cv::Mat og = image.clone();
    double ms = measure_performance([&]() {gaussianBlurCPU(image, generateGaussianKernel(2, 5)); });
    std::cout << "Gaussian blur took " << ms << "ms" << std::endl;

    // Display images
	cv::vconcat(og, image, image);
	imshow("Duck Blurred", image);
	cv::waitKey(0);

	return 0;
}