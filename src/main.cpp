#include<Cl/opencl.hpp>
#include<opencv2/opencv.hpp>
#include<iostream>

struct RGBA {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
};

cv::Mat GaussianBlur(const unsigned char& input, unsigned char& output, int kernelSize, float sigma) {
    for(int i = 0; i < input.rows; i++) {
        for(int j = 0; j < input.cols; j++) {
            RGBA pixel = inputData[i * input.cols + j];
        }
	}
    return output;
}

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
	imshow("Duck", image);
	cv::waitKey(0);
	return 0;
}