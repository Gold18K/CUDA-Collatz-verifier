#pragma once

// Inclusions
#include <cstdint>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace Kernels {

    struct uint128_t {
        uint64_t high;
        uint64_t low;
    };

    __host__ __device__ bool geq(const uint128_t& _a,
                                 const uint128_t& _b);
    __device__ uint128_t collatz_step(const uint128_t& _a,
                                      const uint8_t& _mul,
                                      const uint8_t* _cases,
                                      uint32_t* _checksum,
                                      bool* _skip_task);
    __global__ void collatz(const uint64_t* _start_point,
                            const uint64_t* _n_of_thread_cycles,
                            const uint64_t* _residues,
                            uint64_t* _total_checksum,
                            bool* _skip_task);
	void launch_collatz(const bool& _wait = true);

}
