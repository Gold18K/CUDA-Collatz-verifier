#pragma once

// Inclusions
#include <cstdint>

class Cuda_Utilities {

public:

    // Enum classes
    enum class Unit {
        HOST, DEVICE
    };

    // Static methods
    static void     cuda_choose_device(const bool& _verbose = false);
    static void     cuda_synchronize();
    static void     cuda_device_reset();
    static uint64_t get_used_host_memory();
    static uint64_t get_used_device_memory();
    static bool     get_device_status();
    static void     increase_used_host_memory(const uint64_t& _val);
    static void     increase_used_device_memory(const uint64_t& _val);
	static void     decrease_used_host_memory(const uint64_t& _val);
    static void     decrease_used_device_memory(const uint64_t& _val);

private:

    // Constructor
    Cuda_Utilities() = delete; // Non-instantiable class

    // Static fields
    static bool     device_status;       // False if no device has been choosen yet, or cudaDeviceReset() has been called
    static uint64_t total_host_memory;   // Total host memory allocated in bytes
    static uint64_t total_device_memory; // Total device memory allocated in bytes


};
