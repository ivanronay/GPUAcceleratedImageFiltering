// Naive Gaussian Blur
__kernel void gaussian_blur_horizontal(
    __global const uchar* input,
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
            uchar pixelValue = input[(y * width + clampedindex) * channels + c];
			sums[c] += pixelValue * kernel_buffer[k + radius];
        }
    }
    for (int c = 0; c < channels; ++c) {
		output[(y * width + x) * channels + c] = sums[c];
    }
}

__kernel void gaussian_blur_vertical(
    __global const float* input,
    __global uchar* output,
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
        output[(y * width + x) * channels + c] = sums[c];
    }
}

// Gaussian Blur with shared memory optimization
__kernel void gaussian_blur_horizontal_shared(
    __global const uchar* input,
    __global float* output,
    __constant float* kernel_buffer,
    __local uchar* scratch,
    const int width,
    const int height,
    const int channels,
    const int radius) 
{
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    int gx = get_group_id(0);
    int gy = get_group_id(1);

    int lsize = get_local_size(0);
    int tile_width = lsize + 2 * radius;
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    // copy pixels to shared memory: shared := [_,_,p,p...p,p,_,_]
    for(int c = 0; c < channels; ++c)
        scratch[(ly * tile_width + lx + radius)*channels + c] = (x<width && y<height) ? input[(y*width+x)*channels + c] : 0;

    // copy to left halo, clamp if necessary: shared := [h,h,p,p...p,p,_,_]
    if(lx < radius){
        int clampedindex = clamp(gx * lsize + lx - radius, 0, width - 1);
        for(int c = 0; c < channels; ++c)
            scratch[(ly * tile_width + lx)*channels + c] = input[(y*width+clampedindex)*channels + c];
    }

    // copy to right halo, clamp if necessary: shared := [h,h,p,p...p,p,h,h]
    if(lx > lsize - 1 - radius && lx < lsize){
        int clampedindex = clamp(gx * lsize + lx + radius, 0, width - 1);
        for(int c = 0; c < channels; ++c)
            scratch[(ly * tile_width + lx + 2*radius)*channels + c] = input[(y*width+clampedindex)*channels + c];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // If the thread is outside the image bounds, do nothing
    if (x >= width || y >= height) return;

    float sums[4] = { 0,0,0,0 };
    for (int k =-radius; k <= radius; ++k) {
        for (int c = 0; c < channels; ++c) {
            unsigned char pixelValue = scratch[(ly * tile_width + lx + radius + k) * channels + c];
			sums[c] += pixelValue * kernel_buffer[k + radius];
        }
    }
    for (int c = 0; c < channels; ++c) {
		output[(y * width + x) * channels + c] = sums[c];
    }
}

__kernel void gaussian_blur_vertical_shared(
    __global const float* input,
    __global uchar* output,
    __constant float* kernel_buffer,
    __local float* scratch,
    const int width,
    const int height,
    const int channels,
    const int radius) 
{
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    int gx = get_group_id(0);
    int gy = get_group_id(1);

    int lsize = get_local_size(1); // changes to get_local_size(1) for vertical blur
    int tile_width = get_local_size(0);
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    // copy pixels to shared memory: shared := [_,_,p,p...p,p,_,_]
    for(int c = 0; c < channels; ++c)
        scratch[((ly + radius) * tile_width + lx)*channels + c] = (x<width && y<height) ? input[(y*width+x)*channels + c] : 0;

    // copy to left halo, clamp if necessary: shared := [h,h,p,p...p,p,_,_]
    if(ly < radius){
        int clampedindex = clamp(gy * lsize + ly - radius, 0, height - 1);
        for(int c = 0; c < channels; ++c)
            scratch[(ly * tile_width + lx)*channels + c] = input[(clampedindex*width+x)*channels + c];
    }

    // copy to right halo, clamp if necessary: shared := [h,h,p,p...p,p,h,h]
    if(ly > lsize - 1 - radius && ly < lsize){
        int clampedindex = clamp(gy * lsize + ly + radius, 0, height - 1);
        for(int c = 0; c < channels; ++c)
            scratch[((ly + 2*radius) * tile_width + lx)*channels + c] = input[(clampedindex*width+x)*channels + c];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // If the thread is outside the image bounds, do nothing
    if (x >= width || y >= height) return;

    float sums[4] = { 0,0,0,0 };
    for (int k =-radius; k <= radius; ++k) {
        for (int c = 0; c < channels; ++c) {
            float pixelValue = scratch[((ly + radius + k) * tile_width + lx) * channels + c];
			sums[c] += pixelValue * kernel_buffer[k + radius];
        }
    }
    for (int c = 0; c < channels; ++c) {
		output[(y * width + x) * channels + c] = sums[c];
    }
}