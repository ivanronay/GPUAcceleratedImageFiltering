#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include "cvutils.hpp"
#include "clutils.hpp"
namespace fs = std::filesystem;

std::string selectImage(const std::string& dir) {
    std::vector<std::string> files;
    for (auto& f : fs::directory_iterator(dir)) {
        auto ext = f.path().extension().string();
        if (ext == ".jpg" || ext == ".png" || ext == ".jpeg")
            files.push_back(f.path().string());
    }

    if (files.empty()) {
        std::cerr << "No images found\n";
        return "";
    }

    std::cout << "Pick an image:\n";
    for (int i = 0; i < files.size(); i++)
        std::cout << i << ": " << fs::path(files[i]).filename().string() << "\n";

    int n;
    std::cin >> n;
    if (n < 0 || n >= files.size()) n = 0;
    return files[n];
}

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

int main() 
{
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

    // Initialize OpenCL using the helper
    OpenCLEnv env;
    cl::Program program;
    try {
        env = initOpenCL();
        std::string source = loadKernelSource("../kernels/main.cl");
        buildProgram(program, source, env);
    } catch (const std::exception& e) {
        std::cerr << "OpenCL Initialization Error. " << std::endl;
        return 1;
    }

    // silence irritating OpenCV warnings
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

	std::string imagePath = selectImage("../Assets");
	if (imagePath.empty()) return 1;
	cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
	if (image.empty()) {
		std::cerr << "Failed to load image\n";
		return 1;
	}

	int radius;
	std::cout << "Radius (1-15): ";
	std::cin >> radius;
	if (radius < 1 || radius > 15) {
		std::cerr << "Invalid radius\n";
		return 1;
	}

	cv::Mat noisyImage   = TIME_IT("Salt & Pepper", addSaltAndPepperNoise(image, 0.1));

	std::cout << "\n--- Filter idok (radius=" << radius << ") ---\n";
	cv::Mat medianResult = TIME_IT("Median CPU", medianFilterCPU(noisyImage, radius));

    cv::Mat medianResultGPU = TIME_IT("Median GPU (Naive)", [&]() {
        cv::Mat result = cv::Mat::zeros(noisyImage.rows, noisyImage.cols, noisyImage.type());
        runImageFilter2D(program, "medianFilter", env, noisyImage.data, result.data, 
                   noisyImage.cols, noisyImage.rows, noisyImage.channels(), radius);
        return result;
    }());

    cv::Mat medianResultGPUShared = TIME_IT("Median GPU (Shared)", [&]() {
        cv::Mat result = cv::Mat::zeros(noisyImage.rows, noisyImage.cols, noisyImage.type());
        runImageFilter2DShared(program, "medianFilterShared", env, noisyImage.data, result.data,
                   noisyImage.cols, noisyImage.rows, noisyImage.channels(), radius);
        return result;
    }());

	cv::imshow("Original",                  image);
	cv::imshow("Noisy (10% salt & pepper)", noisyImage);
	cv::imshow("Median filter CPU",         medianResult);
	cv::imshow("Median filter GPU Naive",   medianResultGPU);
	cv::imshow("Median filter GPU Shared",  medianResultGPUShared);
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