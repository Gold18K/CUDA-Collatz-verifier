#include "Cuda_Utilities.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

// Static methods
void   Cuda_Utilities::cuda_choose_device(const bool& _verbose) {
    cudaError_t cudaStatus = cudaSetDevice(0);

    if (_verbose) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        std::cout << "Choosen GPU: " << deviceProp.name << std::endl;
    }

    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaSetDevice FAILED with error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

	device_status = true;
}
void   Cuda_Utilities::cuda_synchronize() {
    cudaError_t cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize FAILED with error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

}
void   Cuda_Utilities::cuda_device_reset() {
    cudaError_t cudaStatus = cudaDeviceReset();

    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceReset FAILED with error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

	device_status = false;
}
uint64_t Cuda_Utilities::get_used_host_memory() {
    return total_host_memory;
}
uint64_t Cuda_Utilities::get_used_device_memory() {
    return total_device_memory;
}
bool   Cuda_Utilities::get_device_status() {
    return device_status;
}
void   Cuda_Utilities::increase_used_host_memory(const uint64_t& _val) {
    total_host_memory += _val;
}
void   Cuda_Utilities::increase_used_device_memory(const uint64_t& _val) {
    total_device_memory += _val;
}
void   Cuda_Utilities::decrease_used_host_memory(const uint64_t& _val) {
    total_host_memory -= _val;
}
void   Cuda_Utilities::decrease_used_device_memory(const uint64_t& _val) {
    total_device_memory -= _val;
}

// Static fields
bool   Cuda_Utilities::device_status     = false;
uint64_t Cuda_Utilities::total_host_memory   = 0;
uint64_t Cuda_Utilities::total_device_memory = 0;
