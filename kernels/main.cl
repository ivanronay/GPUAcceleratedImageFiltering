// Naive Gaussian Blur
__kernel void gaussian_blur_horizontal(
    __global const unsigned char* input,
    __global float* output,
    __constant float* kernel_buffer,
    const int width,
    const int height,
    const int channels,
    const int radius) 
{
    // Threads are launched in 2D (x, y)
    int x = get_global_id(0);
    int y = get_global_id(1);

    // If the thread is outside the image bounds, do nothing
    if (x >= width || y >= height) return; 

    float sums[4] = { 0,0,0,0 };
    for (int k =-radius; k <= radius; ++k) {
		int clampedindex = clamp(x + k, 0, width - 1);
        for (int c = 0; c < channels; ++c) {
            unsigned char pixelValue = input[(y * width + clampedindex) * channels + c];
			sums[c] += pixelValue * kernel_buffer[k + radius];
        }
    }
    for (int c = 0; c < channels; ++c) {
		output[(y * width + x) * channels + c] = sums[c];
    }
}

__kernel void gaussian_blur_vertical(
    __global const float* input,
    __global unsigned char* output,
    __constant float* kernel_buffer,
    const int width,
    const int height,
    const int channels,
    const int radius) 
{
    // Threads are launched in 2D (x, y)
    int x = get_global_id(0);
    int y = get_global_id(1);

    // If the thread is outside the image bounds, do nothing
    if (x >= width || y >= height) return; 

    float sums[4] = {0,0,0,0};
    for (int k =-radius; k <= radius; ++k) {
		int clampedindex = clamp(y + k, 0, height - 1);
        for (int c = 0; c < channels; ++c) {
            float pixelValue = input[(clampedindex * width + x) * channels + c];
            sums[c] += pixelValue * kernel_buffer[k + radius];
        }
    }
    for (int c = 0; c < channels; ++c) {
        // Clamp the result to [0, 255] and convert to unsigned char
        output[(y * width + x) * channels + c] = sums[c];
    }
}