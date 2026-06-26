#define __CL_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>

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

void buildProgram(cl::Program& program, const std::string& source, const OpenCLEnv& env, const char* options = "-cl-std=CL3.0") {
    program = cl::Program(env.context, source);
    try {
        program.build({ env.device }, options);
    } catch (const cl::Error&) {
        std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(env.device) << std::endl;
        throw;
    }
}

void runProgram(cl::Program& program, const std::string& function_name, OpenCLEnv& env, unsigned char* data, int width, int height, int channels) {
    
	size_t size = width * height * channels;
    cl::Buffer input(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, data);
    cl::Buffer output(env.context, CL_MEM_WRITE_ONLY, size);

    cl::KernelFunctor<cl::Buffer, cl::Buffer> kernel(program, function_name);

    kernel(cl::EnqueueArgs(env.queue, cl::NDRange(size)),input, output);
}

// Runs a 2D image processing kernel with the necessary parameters
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

// Runs a 2D image processing kernel using local/shared memory optimization
inline void runImageFilter2DShared(cl::Program& program, const std::string& function_name, OpenCLEnv& env, 
                            unsigned char* input_data, unsigned char* output_data, 
                            int width, int height, int channels, int radius,
                            int local_w = 16, int local_h = 16) {
    
    size_t size = width * height * channels;
    cl::Buffer input(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, input_data);
    cl::Buffer output(env.context, CL_MEM_WRITE_ONLY, size);

    cl::KernelFunctor<cl::Buffer, cl::Buffer, int, int, int, int> kernel(program, function_name);

    size_t gx = roundUp(local_w, width);
    size_t gy = roundUp(local_h, height);

    cl::NDRange global_size(gx, gy);
    cl::NDRange local_size(local_w, local_h);

    try {
        kernel(cl::EnqueueArgs(env.queue, global_size, local_size), input, output, width, height, channels, radius);
        env.queue.enqueueReadBuffer(output, CL_TRUE, 0, size, output_data);
    } catch (const cl::Error&) {
        std::cerr << "Error on runImageFilter2DShared";
        throw;
    }
}