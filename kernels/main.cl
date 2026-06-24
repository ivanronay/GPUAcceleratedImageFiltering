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

unsigned char getMedianHistogram(
    unsigned char* window,
    int size)
{
    int hist[256] = {0};

    // Histogram építése
    for (int i = 0; i < size; ++i)
    {
        hist[window[i]]++;
    }

    // A medián pozíciója
    int target = size / 2;

    int count = 0;

    // Keresés a histogramban
    for (int value = 0; value < 256; ++value)
    {
        count += hist[value];

        if (count > target)
        {
            return (unsigned char)value;
        }
    }

    return 255; // Elméletileg sosem fut le
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
    
    unsigned char window[225]; 

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
        
        // get median using
        output[(y * width + x) * channels + c] = getMedianHistogram(window, windowSize);
    }
}
