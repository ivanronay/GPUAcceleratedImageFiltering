#include <Cl/opencl.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

#define PI 3.14159265358979323846

template <typename Func>
double measure_performance(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Generates 1D gaussian kernel of size 2r+1 with standard deviation sigma
std::vector<float> generateGaussianKernel(int radius, float sigma) {
    std::vector<float> kernel((2 * radius + 1));
    float sum = 0.0f;
    for(int i = -radius; i <= radius; ++i) {
        auto value = std::exp(-(i * i) / (2.0f * sigma * sigma));
        kernel[i + radius] = value;
        sum += value;
    }
    for(float& value : kernel) {
        value /= sum;
    }
    return kernel;
}

// Applies a gaussian blur to src using the given kernel
void gaussianBlurCPU(cv::Mat& src, std::vector<float> kernel) {
    int W = src.cols, H = src.rows, C = src.channels();
	int kernelRadius = kernel.size() / 2;
	std::vector temp = std::vector<unsigned char>(W * H * C);
	auto inputdata = src.data;

    // gaussian blur can be applied separately first row-wise then collumn-wise
    for(int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            for (int c = 0; c < C; ++c) {
                float sum = 0.0f;
                for (int k = -kernelRadius; k <= kernelRadius; ++k) {
					int clampedindex = std::clamp(j + k, 0, W - 1);
                    unsigned char pixelValue = inputdata[(i * W + clampedindex) * C + c];
					sum += pixelValue * kernel[k + kernelRadius];
                }
				temp[(i * W + j) * C + c] = static_cast<unsigned char>(sum);
            }
        }
	}

    for (int j = 0; j < W; ++j) {
        for (int i = 0; i < H; ++i) {
            for (int c = 0; c < C; ++c) {
                float sum = 0.0f;
                for (int k = -kernelRadius; k <= kernelRadius; ++k) {
					int clampedindex = std::clamp(i + k, 0, H - 1);
                    unsigned char pixelValue = temp[(j * H + clampedindex) * C + c];
                    sum += pixelValue * kernel[k + kernelRadius];
                }
                inputdata[(j * H + i) * C + c] = static_cast<unsigned char>(sum);
            }
        }
    }
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