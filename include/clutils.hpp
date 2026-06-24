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
        << "  Device:   " << env.device.getInfo<CL_DEVICE_NAME>() << std::endl;

    return env;
}

inline std::string loadKernelSource(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open kernel file: " + path);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void buildProgram(cl::Program& program, const std::string& source, const OpenCLEnv& env) {
    program = cl::Program(env.context, source);
    try {
        program.build({ env.device });
    } catch (const cl::Error&) {
        std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(env.device) << std::endl;
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