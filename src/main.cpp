#include <Cl/opencl.hpp>
#include <iostream>
#include <chrono>
#include "cvutils.hpp"

#define PI 3.14159265358979323846

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

	cv::Mat image = cv::imread("../assets/duck.jpg", cv::IMREAD_COLOR);

    if(image.empty()) {
        std::cerr << "Could not read the image" << std::endl;
        return 1;
	}

	cv::resize(image, image, cv::Size(512, 512* image.rows/image.cols));
	cv::Mat og = image.clone();
    double ms = measure_performance([&]() {gaussianBlurCPU(image, generateGaussianKernel(2, 5)); });
    std::cout << "This took " << ms << "ms" << std::endl;

    // Display images
	cv::vconcat(og, image, image);
	imshow("Duck Blurred", image);
	cv::waitKey(0);

	return 0;
}