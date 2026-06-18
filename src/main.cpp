#include<Cl/opencl.hpp>
#include<opencv2/opencv.hpp>
#include<iostream>



int main() {
	// check for device support
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (auto& platform : platforms) {
        std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (auto& device : devices) {
            std::cout << "  Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        }
    }

    // silence irritating OpenCV warnings
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

	cv::Mat image = cv::imread("../assets/duck.jpg", cv::IMREAD_COLOR);
	imshow("Duck", image);
	cv::waitKey(0);
	return 0;
}