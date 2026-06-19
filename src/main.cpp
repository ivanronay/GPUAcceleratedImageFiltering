#include<CL/opencl.hpp>
#include<opencv2/opencv.hpp>
#include<algorithm>
#include<cstdlib>
#include<iostream>
#include<vector>

struct RGBA {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
};


cv::Mat gaussianBlurCPU(const cv::Mat& src, int radius, float sigma) {
    int W = src.cols, H = src.rows, C = src.channels();
    cv::Mat dst = cv::Mat::zeros(H, W, src.type());

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float weightSum = 0.0f;
            float vals[3] = { 0, 0, 0 };

            for (int ky = -radius; ky <= radius; ++ky) {
                for (int kx = -radius; kx <= radius; ++kx) {
                    // Gaussian weight
                    float w = std::exp(-(kx*kx + ky*ky) / (2.0f * sigma * sigma));

                    // Clamp to border
                    int px = std::max(0, std::min(W - 1, x + kx));
                    int py = std::max(0, std::min(H - 1, y + ky));

                    const uchar* row = src.ptr<uchar>(py);
                    for (int c = 0; c < C; ++c)
                        vals[c] += row[px * C + c] * w;

                    weightSum += w;
                }
            }

            uchar* out = dst.ptr<uchar>(y);
            for (int c = 0; c < C; ++c)
                out[x * C + c] = static_cast<uchar>(vals[c] / weightSum);
        }
    }

    return dst;
}

// ---------------------------------------------------------------------------
// CPU reference: Median filter
// Applies an (2*radius+1) x (2*radius+1) median filter independently to
// each colour channel. Border pixels are clamped to the image edge.
// ---------------------------------------------------------------------------
cv::Mat medianFilterCPU(const cv::Mat& src, int radius) {
    int W = src.cols, H = src.rows, C = src.channels();
    cv::Mat dst = cv::Mat::zeros(H, W, src.type());

    int windowSize = (2 * radius + 1) * (2 * radius + 1);
    std::vector<uchar> window(windowSize);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            uchar* out = dst.ptr<uchar>(y);

            for (int c = 0; c < C; ++c) {
                int idx = 0;
                // For each pixel
                for (int ky = -radius; ky <= radius; ++ky) {
                    for (int kx = -radius; kx <= radius; ++kx) {
                        int px = std::max(0, std::min(W - 1, x + kx));
                        int py = std::max(0, std::min(H - 1, y + ky));
                        window[idx++] = src.ptr<uchar>(py)[px * C + c];
                    }
                }
                std::sort(window.begin(), window.end());
                out[x * C + c] = window[windowSize / 2];
            }
        }
    }

    return dst;
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
	const int radius = 3; // 7x7 window
	cv::Mat medianResult = medianFilterCPU(image, radius);

	cv::imshow("Original", image);
	cv::imshow("Median filter (radius=3)", medianResult);
	cv::waitKey(0);
	return 0;
}