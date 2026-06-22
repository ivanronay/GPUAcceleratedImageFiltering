// No built-in sort function, so we use bubble sort
// Max 49 elements with 3x3 window radius should be fine
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
    
    // Fixed size array for the window elements
    // 225 elements are enough for radius = 7 (15x15 window)
    unsigned char window[225]; 

    // Filtering is done separately for each channel (e.g. R, G, B)
    for (int c = 0; c < channels; ++c) {
        int idx = 0;
        
        // Iterate over the window
        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                // Clamping to edge
                int px = clamp(x + kx, 0, width - 1);
                int py = clamp(y + ky, 0, height - 1);
                
                window[idx++] = input[(py * width + px) * channels + c];
            }
        }
        
        // get median
        output[(y * width + x) * channels + c] = getMedian(window, windowSize);
    }
}
