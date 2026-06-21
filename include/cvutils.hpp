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