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

void buildProgram(cl::Program& program, const std::string& source, const OpenCLEnv& env) {
    program = cl::Program(env.context, source);
    try {
        program.build({ env.device });
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