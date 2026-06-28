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

// let user pick an image from the given dir
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

template <typename Func>
double measure_performance(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
	// init opencl context and build kernels
	OpenCLEnv env;
	cl::Program medianProgram;
	cl::Program gaussianProgram;
	try {
		env = initOpenCL();
		std::string medianSource = loadKernelSource("../kernels/median.cl");
		std::string gaussianSource = loadKernelSource("../kernels/gaussian.cl");

		buildProgram(medianProgram, medianSource, env);
		buildProgram(gaussianProgram, gaussianSource, env);
	}
	catch (const std::exception& e) {
		std::cerr << "OpenCL Initialization Error. " << std::endl;
		std::cerr << e.what() << std::endl;
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
	cv::resize(image, image, cv::Size(1024, 1024*image.rows/image.cols));

	int radius = 2;
	int sigma = 5;
	bool useUnOptimized = false;
	std::vector<float> gaussianKernel = generateGaussianKernel(radius,sigma);
	cv::Mat result = cv::Mat::zeros(image.rows, image.cols, image.type());
	cv::Mat result2 = cv::Mat::zeros(image.rows, image.cols, image.type());
	cv::Mat noisyImage;
	cv::Mat medianResult;

	std::cout << "=====================================" << std::endl;
	std::cout << "Press" \
		      << "\t g: Gaussian Blur" << std::endl \
			  << "\t m: Median Filter" << std::endl \
			  << "\t o: Show original image" << std::endl \
			  << "\t +: Increase radius" << std::endl \
			  << "\t -: Decrease radius" << std::endl \
			  << "\t q or ESC: Quit" << std::endl
		      << "\t u: Toggle unoptimized versions" << std::endl
		      << "\t b: Benchmark median filter (R 1-15) -> CSV" << std::endl
		      << "\t n: Benchmark gaussian blur (R 1-15) -> CSV" << std::endl;
	std::cout << "=====================================" << std::endl << std::endl;

	cv::imshow("Display", image);
	
	// interactive event loop
	while (true) {
		int key = cv::waitKey(0) & 0xFF;
		
		// Uncomment this if you want to see what keycode OpenCV is receiving:
		std::cout << "Key pressed: " << key << " (char: " << (char)key << ")" << std::endl;

		if(key == 27 || key == 'q') {
			break;
		}
		else if (key == 'o') {
			cv::imshow("Display", image);
		}
		else if(key == 'g') {
			std::cout << "=====================================" << std::endl;
			if (useUnOptimized) {
				double msCPU = measure_performance([&]() {gaussianBlurCPU(image, result, gaussianKernel); });
				double msGPU = measure_performance([&]() {runGaussianBlurGPU(gaussianProgram, env, image.data, result.data, gaussianKernel, image.cols, image.rows, image.channels(), radius); });
				std::cout << "Gaussian blur (CPU) took " << msCPU << "ms" << std::endl;
				std::cout << "Gaussian blur (GPU) took " << msGPU << "ms" << std::endl;
			}
			double msGPU2 = measure_performance([&]() {runGaussianBlurGPUshared(gaussianProgram, env, image.data, result.data, gaussianKernel, image.cols, image.rows, image.channels(), radius); });
			std::cout << "Gaussian blur (GPU shared) took " << msGPU2 << "ms" << std::endl;
			std::cout << "=====================================" << std::endl << std::endl;
			cv::imshow("Display", result);
		}
		else if (key == 'm') {
			std::cout << "=====================================" << std::endl;
			if (useUnOptimized) {
				double msCPU = measure_performance([&]() {result = medianFilterCPU(image, radius);});
				double msGPU = measure_performance([&]() {runImageFilter2D(medianProgram, "medianFilter", env, image.data, result.data, image.cols, image.rows, image.channels(), radius); });
				std::cout << "Median filter (CPU) took " << msCPU << "ms" << std::endl;
				std::cout << "Median filter (GPU) took " << msGPU << "ms" << std::endl;
			}
			double msGPU2 = measure_performance([&]() {runImageFilter2DShared(medianProgram,"medianFilterShared", env, image.data, result.data, image.cols, image.rows, image.channels(), radius); });
			std::cout << "Median filter (GPU shared) took " << msGPU2 << "ms" << std::endl;
			std::cout << "=====================================" << std::endl << std::endl;

			cv::imshow("Display", result);
		}
		else if (key == 'p') {
			double SPms = measure_performance([&]() {noisyImage = addSaltAndPepperNoise(image, 0.1); });
			double CPUms = measure_performance([&]() {medianResult = medianFilterCPU(noisyImage, radius); });
			double GPUms = measure_performance([&]() {
				runImageFilter2D(medianProgram, "medianFilter", env, noisyImage.data, result.data,
					noisyImage.cols, noisyImage.rows, noisyImage.channels(), radius);
				});

			double GPU2ms = measure_performance([&]() {
				runImageFilter2DShared(medianProgram, "medianFilterShared", env, noisyImage.data, result2.data,
					noisyImage.cols, noisyImage.rows, noisyImage.channels(), radius);
				});

			cv::Mat l_orig = addLabel(image, "Original");
			cv::Mat l_noisy = addLabel(noisyImage, "Noisy");
			cv::Mat l_cpu = addLabel(medianResult, "CPU");
			cv::Mat l_naive = addLabel(result, "GPU Naive");
			cv::Mat l_shared = addLabel(result2, "GPU Shared");
			cv::Mat l_blank = cv::Mat::zeros(l_orig.rows, l_orig.cols, l_orig.type());

			cv::Mat row1, row2, grid;
			cv::hconcat(std::vector<cv::Mat>{l_orig, l_noisy, l_cpu}, row1);
			cv::hconcat(std::vector<cv::Mat>{l_naive, l_shared, l_blank}, row2);
			cv::vconcat(row1, row2, grid);

			cv::imshow("Display", grid);
		}
		else if (key == '+') {
			radius = std::min(15, radius + 1);
			gaussianKernel = generateGaussianKernel(radius, sigma);
			std::cout << "Radius increased to " << radius << std::endl;
		}
		else if (key == '-') {
			radius = std::max(1, radius - 1);
			gaussianKernel = generateGaussianKernel(radius, sigma);
			std::cout << "Radius decreased to " << radius << std::endl;
		}
		else if (key == 'u') {
			useUnOptimized = !useUnOptimized;
			std::cout << "Use unoptimized versions: " << (useUnOptimized ? "ON" : "OFF") << std::endl;
		}
		else if (key == 'b') {
			std::cout << "Running benchmark (Radius 1 to 15)..." << std::endl;
			std::ofstream csv("../analysis/benchmark_results.csv");
			if (!csv.is_open()) {
				std::cerr << "Failed to open CSV for writing!" << std::endl;
			} else {
				csv << "Radius,CPU_ms,GPU_Naive_ms,GPU_Shared_ms\n";
				const int numRuns = 5;
				cv::Mat noisyBench = addSaltAndPepperNoise(image, 0.1);
				for (int r = 1; r <= 15; ++r) {
					std::cout << "  r=" << r << "..." << std::endl;
					double msCPU    = measure_performance([&]() { medianFilterCPU(noisyBench, r); });
					double msGPU = 0.0, msShared = 0.0;
					for (int i = 0; i < numRuns; i++) {
						msGPU    += measure_performance([&]() { runImageFilter2D(medianProgram, "medianFilter", env, noisyBench.data, result.data, noisyBench.cols, noisyBench.rows, noisyBench.channels(), r); });
						msShared += measure_performance([&]() { runImageFilter2DShared(medianProgram, "medianFilterShared", env, noisyBench.data, result2.data, noisyBench.cols, noisyBench.rows, noisyBench.channels(), r); });
					}
					msGPU /= numRuns; msShared /= numRuns;
					csv << r << "," << msCPU << "," << msGPU << "," << msShared  << "\n";
				}
				csv.close();
				std::cout << "Done. Saved to ../analysis/median_benchmark.csv" << std::endl;
			}
		}
		else if (key == 'n') {
			std::cout << "Running Gaussian benchmark (Radius 1 to 15)..." << std::endl;
			std::ofstream csv("../analysis/gaussian_benchmark.csv");
			if (!csv.is_open()) {
				std::cerr << "Failed to open CSV for writing!" << std::endl;
			} else {
				csv << "Radius,CPU_ms,GPU_Naive_ms,GPU_Shared_ms\n";
				const int numRuns = 10;
				for (int r = 1; r <= 15; ++r) {
					std::cout << "  r=" << r << "..." << std::endl;
					float msCPUsum = 0.0f; float msGPUsum = 0.0f; float msSharedsum = 0.0f;
					for (int i = 0; i < numRuns; ++i)
					{
						std::vector<float> gk = generateGaussianKernel(r, (float)r / 2.0f + 0.5f);
						double msCPU = measure_performance([&]() { gaussianBlurCPU(image, result, gk); });
						double msGPU = measure_performance([&]() { runGaussianBlurGPU(gaussianProgram, env, image.data, result.data, gk, image.cols, image.rows, image.channels(), r); });
						double msShared = measure_performance([&]() { runGaussianBlurGPUshared(gaussianProgram, env, image.data, result2.data, gk, image.cols, image.rows, image.channels(), r); });
						msCPUsum += msCPU; msGPUsum += msGPU; msSharedsum += msShared;
					}
					msCPUsum /= numRuns; msGPUsum /= numRuns; msSharedsum /= numRuns;
					csv << r << "," << msCPUsum << "," << msGPUsum << "," << msSharedsum << "\n";
				}
				csv.close();
				std::cout << "Done. Saved to ../analysis/gaussian_benchmark.csv" << std::endl;
			}
		}
	}
	return 0;
}