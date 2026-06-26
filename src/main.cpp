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
	std::vector<float> gkernel = generateGaussianKernel(radius,sigma);
	cv::Mat result = cv::Mat::zeros(image.rows, image.cols, image.type());

	std::cout << "=====================================" << std::endl;
	std::cout << "Press" \
		      << "\t g: Gaussian Blur" << std::endl \
			  << "\t m: Median Filter" << std::endl \
			  << "\t o: Show original image" << std::endl \
			  << "\t +: Increase radius" << std::endl \
			  << "\t -: Decrease radius" << std::endl \
			  << "\t q or ESC: Quit" << std::endl;
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
			double msCPU = measure_performance([&]() {gaussianBlurCPU(image, result, gkernel); });
			double msGPU = measure_performance([&]() {runGaussianBlurGPU(program, env, image.data, result.data, gkernel, image.cols, image.rows, image.channels(), radius); });
			double msGPU2 = measure_performance([&]() {runGaussianBlurGPUshared(program, env, image.data, result.data, gkernel, image.cols, image.rows, image.channels(), radius); });
			std::cout << "=====================================" << std::endl;
			std::cout << "Gaussian blur (CPU) took " << msCPU << "ms" << std::endl;
			std::cout << "Gaussian blur (GPU) took " << msGPU << "ms" << std::endl;
			std::cout << "Gaussian blur (GPU shared) took " << msGPU2 << "ms" << std::endl;
			std::cout << "=====================================" << std::endl << std::endl;

			cv::imshow("Display", result);
		}
		else if (key == 'm') {
			double msCPU = measure_performance([&]() {result = medianFilterCPU(image, radius);});
			//double msGPU = measure_performance([&]() {runMedianFilterGPU(program, env, image.data, result.data, image.cols, image.rows, image.channels(), radius); });
			//double msGPU2 = measure_performance([&]() {runMedianFilterGPUshared(program, env, image.data, result.data, image.cols, image.rows, image.channels(), radius); });
			std::cout << "=====================================" << std::endl;
			std::cout << "Median filter (CPU) took " << msCPU << "ms" << std::endl;
			std::cout << "Median filter (GPU) took " << "N/A" << "ms" << std::endl;
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
	}


	return 0;
}