#define __CL_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <iostream>

struct OpenCLEnv {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
};

OpenCLEnv initOpenCL() {
    OpenCLEnv env;

    // Getplatforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found");
    env.platform = platforms.front();

    // Get devices
    std::vector<cl::Device> devices;
    env.platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) throw std::runtime_error("No GPU found");
    env.device = devices.front();

    // Context and Queue
    env.context = cl::Context(env.device);
    env.queue = cl::CommandQueue(env.context, env.device);

    std::cout << "OpenCL Initialized:" << std::endl
        << "  Platform: " << env.platform.getInfo<CL_PLATFORM_NAME>() << std::endl
		<< "\t version: " << env.platform.getInfo<CL_PLATFORM_VERSION>() << std::endl
        << "  Device:   " << env.device.getInfo<CL_DEVICE_NAME>() << std::endl;

    return env;
}

inline std::string loadKernelSource(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open kernel file: " + path);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void buildProgram(cl::Program& program, const std::string& source, const OpenCLEnv& env, const char* options = "-cl-std=CL2.0") {
    program = cl::Program(env.context, source);
    try {
        program.build({ env.device }, options);
    } catch (const cl::Error&) {
        std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(env.device) << std::endl;
        throw;
    }
}

// naive 2d kernel execution
void runImageFilter2D(cl::Program& program, const std::string& function_name, OpenCLEnv& env, 
                      unsigned char* input_data, unsigned char* output_data, 
                      int width, int height, int channels, int radius) {
    
    size_t size = width * height * channels;
    cl::Buffer input(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, input_data);
    cl::Buffer output(env.context, CL_MEM_WRITE_ONLY, size);

    cl::KernelFunctor<cl::Buffer, cl::Buffer, int, int, int, int> kernel(program, function_name);

    kernel(cl::EnqueueArgs(env.queue, cl::NDRange(width, height)), input, output, width, height, channels, radius);

    env.queue.enqueueReadBuffer(output, CL_TRUE, 0, size, output_data);
}

inline size_t roundUp(size_t group_size, size_t global_size) {
    size_t r = global_size % group_size;
    if (r == 0) return global_size;
    return global_size + group_size - r;
}

// shared memory kernel execution with automatic padding
inline void runImageFilter2DShared(cl::Program& program, const std::string& function_name, OpenCLEnv& env, 
                            unsigned char* input_data, unsigned char* output_data, 
                            int width, int height, int channels, int radius,
                            int local_w = 16, int local_h = 16) {
    
    size_t size = width * height * channels * sizeof(unsigned char);
	size_t local_size = (local_w + 2 * radius) * (local_h + 2 * radius) * sizeof(unsigned char);
    cl::Buffer input(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, input_data);
    cl::Buffer output(env.context, CL_MEM_WRITE_ONLY, size);

    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, int, int, int, int> kernel(program, function_name);

    size_t gx = roundUp(local_w, width);
    size_t gy = roundUp(local_h, height);

    cl::NDRange global(gx, gy);
    cl::NDRange local(local_w, local_h);

    try {
        kernel(cl::EnqueueArgs(env.queue, cl::NullRange, global, local), input, output, cl::Local(local_size), width, height, channels, radius);
        env.queue.enqueueReadBuffer(output, CL_TRUE, 0, size, output_data);
    } catch (const cl::Error&) {
        std::cerr << "Error on runImageFilter2DShared";
        throw;
    }
}

// Runs a 2D image processing kernel with the necessary parameters
void runGaussianBlurGPU(cl::Program& program, OpenCLEnv& env,
    unsigned char* input_data, unsigned char* output_data, std::vector<float>& kernel,
    int width, int height, int channels, int radius) {

    size_t ucharsize = width * height * channels * sizeof(unsigned char);
    size_t floatSize = width * height * channels * sizeof(float);

    cl::Buffer input(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ucharsize, input_data);
    cl::Buffer temp(env.context, CL_MEM_READ_WRITE, floatSize);
    cl::Buffer output(env.context, CL_MEM_WRITE_ONLY, ucharsize);
    cl::Buffer kernelBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kernel.size() * sizeof(float), kernel.data());

    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int> horizontal(program, "gaussian_blur_horizontal");
    horizontal(cl::EnqueueArgs(env.queue, cl::NDRange(width, height)), input, temp, kernelBuffer, width, height, channels, radius);
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int> vertical(program, "gaussian_blur_vertical");
    vertical(cl::EnqueueArgs(env.queue, cl::NDRange(width, height)), temp, output, kernelBuffer, width, height, channels, radius);
    env.queue.enqueueReadBuffer(output, CL_TRUE, 0, ucharsize, output_data);
}

void runGaussianBlurGPUshared(cl::Program& program, OpenCLEnv& env,
    unsigned char* input_data, unsigned char* output_data, std::vector<float>& kernel,
    int width, int height, int channels, int radius) {

    size_t ucharsize = width * height * channels * sizeof(unsigned char);
    size_t floatSize = width * height * channels * sizeof(float);

    cl::Buffer input(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ucharsize, input_data);
    cl::Buffer temp(env.context, CL_MEM_READ_WRITE, floatSize);
    cl::Buffer output(env.context, CL_MEM_WRITE_ONLY, ucharsize);
    cl::Buffer kernelBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kernel.size() * sizeof(float), kernel.data());

    size_t localSize = 16;
    size_t globalSizeX = ((width + localSize - 1) / localSize) * localSize;
    size_t globalSizeY = ((height + localSize - 1) / localSize) * localSize;
    size_t ScracthSizeX = (localSize + 2 * radius) * localSize * channels * sizeof(unsigned char);
    size_t ScracthSizeY = (localSize + 2 * radius) * localSize * channels * sizeof(float);

    cl::NDRange global(globalSizeX, globalSizeY);
    cl::NDRange local(localSize, localSize);

    cl::Kernel horizontal(program, "gaussian_blur_horizontal_shared");
    horizontal.setArg(0, input);
    horizontal.setArg(1, temp);
    horizontal.setArg(2, kernelBuffer);
    horizontal.setArg(3, cl::Local(ScracthSizeX));
    horizontal.setArg(4, width);
    horizontal.setArg(5, height);
    horizontal.setArg(6, channels);
    horizontal.setArg(7, radius);
    env.queue.enqueueNDRangeKernel(horizontal, cl::NullRange, global, local);

    cl::Kernel vertical(program, "gaussian_blur_vertical_shared");
    vertical.setArg(0, temp);
    vertical.setArg(1, output);
    vertical.setArg(2, kernelBuffer);
    vertical.setArg(3, cl::Local(ScracthSizeY));
    vertical.setArg(4, width);
    vertical.setArg(5, height);
    vertical.setArg(6, channels);
    vertical.setArg(7, radius);
    env.queue.enqueueNDRangeKernel(vertical, cl::NullRange, global, local);

    env.queue.enqueueReadBuffer(output, CL_TRUE, 0, ucharsize, output_data);
}