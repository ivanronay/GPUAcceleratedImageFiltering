#define MAX_RAD 15
#define MAX_WINDOW_SIZE ((2 * MAX_RAD + 1) * (2 * MAX_RAD + 1))

unsigned char getMedian(unsigned char* window, int size) {
    for (int i = 0; i < size - 1; ++i) {
        for (int j = 0; j < size - i - 1; ++j) {
            if (window[j] > window[j + 1]) {
                unsigned char temp = window[j];
                window[j] = window[j + 1];
                window[j + 1] = temp;
            }
        }
    }
    return window[size / 2];
}


unsigned char getMedianHistogram(
    unsigned char* window,
    int size)
{
    int hist[256] = {0};
    for (int i = 0; i < size; ++i)
    {
        hist[window[i]]++;
    }
    int target = size / 2;
    int count = 0;
    for (int value = 0; value < 256; ++value)
    {
        count += hist[value];
        if (count > target)
        {
            return (unsigned char)value;
        }
    }

    return 255;
}

// autoselecting between the two medians
unsigned char getMedianAuto(unsigned char* window, int windowSize, int radius) {
    if (radius > 4)
        return getMedianHistogram(window, windowSize);
    else
        return getMedian(window, windowSize);
}

// Naive median filter kernel
__kernel void medianFilter(
    __global const unsigned char* input,
    __global unsigned char* output,
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

    int windowSize = (2 * radius + 1) * (2 * radius + 1);
    
    unsigned char window[MAX_WINDOW_SIZE];

    for (int c = 0; c < channels; ++c) {
        int idx = 0;
        
        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                // Clamping to edge
                int px = clamp(x + kx, 0, width - 1);
                int py = clamp(y + ky, 0, height - 1);
                
                window[idx++] = input[(py * width + px) * channels + c];
            }
        }
        
        // radius > 4: histogram version
        output[(y * width + x) * channels + c] = getMedianAuto(window, windowSize, radius);
    }
}


// Macros for shared memory version
#define WG_W 16
#define WG_H 16

__kernel void medianFilterShared(
    __global const unsigned char* input,
    __global unsigned char* output,
    const int width,
    const int height,
    const int channels,
    const int radius) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    __local unsigned char local_mem[(WG_W + 2 * MAX_RAD) * (WG_H + 2 * MAX_RAD)];
    
    int local_width = get_local_size(0) + 2 * radius;
    int total_elements = local_width * (get_local_size(1) + 2 * radius); 
    
    int total_threads = get_local_size(0) * get_local_size(1);
    int tid = ly * get_local_size(0) + lx;
    
    int group_x = x - lx;
    int group_y = y - ly;

    for (int c = 0; c < channels; ++c) {
        
        for (int i = tid; i < total_elements; i += total_threads) {
            
            int l_row = i / local_width;
            int l_col = i % local_width;
            
            int g_col = group_x - radius + l_col;
            int g_row = group_y - radius + l_row;

            // clamping
            g_col = clamp(g_col, 0, width - 1);
            g_row = clamp(g_row, 0, height - 1);
            
            // loading pixels to local memory
            local_mem[i] = input[(g_row * width + g_col) * channels + c];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (x < width && y < height) {
            int windowSize = (2 * radius + 1) * (2 * radius + 1);
            unsigned char window[MAX_WINDOW_SIZE]; 
            
            int idx = 0;
            // double loop on local tile
            for (int ky = 0; ky <= 2 * radius; ++ky) {
                for (int kx = 0; kx <= 2 * radius; ++kx) {
                    window[idx++] = local_mem[(ly + ky) * local_width + (lx + kx)];
                }
            }
            
            output[(y * width + x) * channels + c] = getMedianAuto(window, windowSize, radius);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
    }
}