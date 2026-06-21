#pragma once
#include <opencv2/opencv.hpp>

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
	std::vector temp = std::vector<float>(W * H * C);
	auto inputdata = src.data;

    // gaussian blur can be applied separately first row-wise then column-wise
    for(int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            float sums[4] = { 0,0,0,0 };
            for (int k = -kernelRadius; k <= kernelRadius; ++k) {
				int clampedindex = std::clamp(j + k, 0, W - 1);
                for (int c = 0; c < C; ++c) {
                    unsigned char pixelValue = inputdata[(i * W + clampedindex) * C + c];
					sums[c] += pixelValue * kernel[k + kernelRadius];
                }
            }
            for (int c = 0; c < C; ++c) {
			    temp[(i * W + j) * C + c] = sums[c];
            }
        }
	}

    for (int j = 0; j < W; ++j) {
        for (int i = 0; i < H; ++i) {
            float sums[4] = {0,0,0,0};
            for (int k = -kernelRadius; k <= kernelRadius; ++k) {
				int clampedindex = std::clamp(i + k, 0, H - 1);
                for (int c = 0; c < C; ++c) {
                    float pixelValue = temp[(clampedindex * W + j) * C + c];
                    sums[c] += pixelValue * kernel[k + kernelRadius];
                }
            }
            for (int c = 0; c < C; ++c) {
                inputdata[(i * W + j) * C + c] = static_cast<unsigned char>(sums[c]);
            }
        }
    }
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

cv::Mat addSaltAndPepperNoise(const cv::Mat& src, double density) {
    cv::Mat dst = src.clone();
    int total = dst.rows * dst.cols;
    int noisyPixels = static_cast<int>(total * density);

    for (int i = 0; i < noisyPixels; ++i) {
        int row = rand() % dst.rows;
        int col = rand() % dst.cols;
        uchar val = (i % 2 == 0) ? 255 : 0;
        uchar* pixel = dst.ptr<uchar>(row) + col * dst.channels();
        for (int c = 0; c < dst.channels(); ++c)
            pixel[c] = val;
    }
    return dst;
}