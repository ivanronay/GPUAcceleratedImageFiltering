#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "cvutils.hpp"
#include "clutils.hpp"

template <typename Func>
double measure_performance(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
	OpenCLEnv env;
	cl::Program program;
	try {
		env = initOpenCL();
		std::string source = loadKernelSource("../kernels/main.cl");
		buildProgram(program, source, env);
	}
	catch (const std::exception& e) {
		std::cerr << "OpenCL Initialization Error. " << std::endl;
		return 1;
	}

    // silence irritating OpenCV warnings
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

	cv::Mat image = cv::imread("../Assets/duck.jpg", cv::IMREAD_COLOR);
	if (image.empty()) {
		std::cerr << "Failed to load image\n";
		return 1;
	}
	cv::resize(image, image, cv::Size(1024, 1024*image.rows/image.cols));

	int radius = 2;
	int sigma = 5;
	bool useUnOptimized = false;
	std::vector<float> gkernel = generateGaussianKernel(radius,sigma);
	cv::Mat result = cv::Mat::zeros(image.rows, image.cols, image.type());

	std::cout << "=====================================" << std::endl;
	std::cout << "Press" \
		      << "\t g: Gaussian Blur" << std::endl \
			  << "\t m: Median Filter" << std::endl \
			  << "\t o: Show original image" << std::endl \
			  << "\t +: Increase radius" << std::endl \
			  << "\t -: Decrease radius" << std::endl \
			  << "\t q or ESC: Quit" << std::endl
		      << "\t u: Toggle unoptimized versions" << std::endl;
	std::cout << "=====================================" << std::endl << std::endl;

	cv::imshow("Display", image);
	while (true) {
		int key = cv::waitKey(0);

		if(key == 27 || key == 'q') {
			break;
		}
		else if (key == 'o') {
			cv::imshow("Display", image);
		}
		else if(key == 'g') {
			std::cout << "=====================================" << std::endl;
			if (useUnOptimized) {
				double msCPU = measure_performance([&]() {gaussianBlurCPU(image, result, gkernel); });
				double msGPU = measure_performance([&]() {runGaussianBlurGPU(program, env, image.data, result.data, gkernel, image.cols, image.rows, image.channels(), radius); });
				std::cout << "Gaussian blur (CPU) took " << msCPU << "ms" << std::endl;
				std::cout << "Gaussian blur (GPU) took " << msGPU << "ms" << std::endl;
			}
			double msGPU2 = measure_performance([&]() {runGaussianBlurGPUshared(program, env, image.data, result.data, gkernel, image.cols, image.rows, image.channels(), radius); });
			std::cout << "Gaussian blur (GPU shared) took " << msGPU2 << "ms" << std::endl;
			std::cout << "=====================================" << std::endl << std::endl;

			cv::imshow("Display", result);
		}
		else if (key == 'm') {
			std::cout << "=====================================" << std::endl;
			if (useUnOptimized) {
				double msCPU = measure_performance([&]() {result = medianFilterCPU(image, radius);});
				// double msGPU = measure_performance([&]() {runMedianFilterGPU(program, env, image.data, result.data, image.cols, image.rows, image.channels(), radius); });
				std::cout << "Median filter (CPU) took " << msCPU << "ms" << std::endl;
				std::cout << "Median filter (GPU) took " << "N/A" << "ms" << std::endl;
			}
			//double msGPU2 = measure_performance([&]() {runMedianFilterGPUshared(program, env, image.data, result.data, image.cols, image.rows, image.channels(), radius); });
			std::cout << "Median filter (GPU shared) took " << "N/A" << "ms" << std::endl;
			std::cout << "=====================================" << std::endl << std::endl;

			cv::imshow("Display", result);
		}
		else if (key == '+') {
			radius = std::min(10, radius + 1);
			gkernel = generateGaussianKernel(radius, sigma);
			std::cout << "Radius increased to " << radius << std::endl;
		}
		else if (key == '-') {
			radius = std::max(1, radius - 1);
			gkernel = generateGaussianKernel(radius, sigma);
			std::cout << "Radius decreased to " << radius << std::endl;
		}
		else if (key == 'u') {
			useUnOptimized = !useUnOptimized;
			std::cout << "Use unoptimized versions: " << (useUnOptimized ? "ON" : "OFF") << std::endl;
		}
	}


	return 0;
}