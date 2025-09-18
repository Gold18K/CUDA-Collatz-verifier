
// Inclusions
#include <algorithm>
#include <chrono>
#include <cstdint>
#include "cuda_runtime.h"
#include "Cuda_Utilities.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <iostream>
#include "Kernels.cuh"
#include <math.h>
#include "Memory.h"

__host__ __device__ bool Kernels::geq(const uint128_t& _a,
                                      const uint128_t& _b) {
    return (_a.high > _b.high) || ((_a.high == _b.high) && (_a.low >= _b.low));
}
__device__ Kernels::uint128_t Kernels::collatz_step(const uint128_t& _a,
                                                    const uint8_t& _mul,
                                                    const uint8_t* _cases,
                                                    uint32_t* _checksum,
                                                    bool* _skip_task) {
    const uint8_t shift = ((_mul > 0) * 3 + (_mul == 0));

    if (__clzll(_a.high) < shift)
        *_skip_task = true;

    uint128_t result = { (_a.high << shift) | (_a.low >> (64 - shift)), (_a.low << shift) + _cases[_mul] };

    result.high += result.low < _cases[_mul];

    if (result.high < (result.low < _cases[_mul]))
        *_skip_task = true;

    result.low  += _a.low;
    result.high += (result.low < _a.low) + _a.high;

    if (result.high < (result.low < _a.low) + _a.high)
        *_skip_task = true;

    atomicAdd(reinterpret_cast<unsigned int*>(_checksum), (shift >> 1) + 1);

    uint8_t tzs = __clzll(__brevll(result.low));

    if (tzs == 64) {
        result.low  = result.high >> __clzll(__brevll(result.high));
        result.high = 0;

        return result;
    }

    result.low  =   (result.low >> tzs) | (result.high << (64 - tzs));
    result.high >>= tzs;

    return result;
}
__global__ void Kernels::collatz(const uint64_t* _start_point,
                                 const uint64_t* _n_of_thread_cycles,
                                 const uint64_t* _residues,
                                 uint64_t* _total_checksum,
                                 bool* _skip_task) {
    const uint64_t global_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id >= 23642078ull)
        return;

    __shared__ uint64_t local_residues[512];
    __shared__ uint32_t block_checksum;
   
    if (threadIdx.x == 0)
        block_checksum = 0;

	local_residues[threadIdx.x] = _residues[512 * blockIdx.x + threadIdx.x];

    __syncthreads();

	const uint64_t start_point        = *_start_point;
    const uint64_t n_of_thread_cycles = *_n_of_thread_cycles;
    const uint8_t  accel_cases[5]     = { 1, 5, 7, 0, 11 };

    for (uint64_t i = start_point; i != start_point + n_of_thread_cycles; ++i) {
		uint128_t n = { i >> 33, 2147483648ull * i + local_residues[threadIdx.x] }; // 64 - 31, 2^31
    
        if (((n.high % 3) + (n.low % 3)) % 3 == 2)
            continue;

        const uint8_t r = (7 * (n.high % 9) + (n.low % 9)) % 9;

        if (r == 2 || r == 4 || r == 8)
            continue;

        uint128_t test = n;

        while (geq(test, n) && !*_skip_task) {
            const uint8_t bitmap = ((test.low & 3) == 3) | (((test.low & 7) == 1) << 1) | (((test.low & 15) == 13) << 2);

            test = Kernels::collatz_step(test, bitmap, accel_cases, &block_checksum, _skip_task);
        }

    }

	__syncthreads();

	if (threadIdx.x == 0)
        atomicAdd(reinterpret_cast<unsigned long long int*>(_total_checksum), block_checksum);

}
void Kernels::launch_collatz(const bool& _wait) {
    constexpr uint32_t n_of_threads   = 512;
	constexpr uint8_t  collatz_sieve  = 31;
    constexpr uint32_t n_of_survivors = 23642078;

    Memory<uint64_t> start_point        = Memory<uint64_t>(1);
    Memory<uint64_t> n_of_thread_cycles = Memory<uint64_t>(1);
    Memory<uint64_t> checksum           = Memory<uint64_t>(1);
	Memory<uint64_t> residues           = Memory<uint64_t>(n_of_survivors);
    Memory<bool>     skip_task          = Memory<bool>(1);

    start_point[0]        = 1ull << 40; // Start testing from 2^70
	n_of_thread_cycles[0] = 1ull << 9;
	checksum[0]           = 0;
	skip_task[0]          = false;

	std::ifstream file("collatz_residues_" + std::to_string(collatz_sieve) + ".txt");

	if (file.is_open()) {
        std::string row = "";
        uint32_t    i   = 0;

        while (std::getline(file, row)) {
			residues[i++] = std::stoull(row);
		}

		file.close();
	}

    else {
        std::cerr << "File with residues not found!" << std::endl;
        std::exit(-1);
    }

	Memory<uint64_t>::copy(start_point,        Cuda_Utilities::Unit::DEVICE, start_point,        Cuda_Utilities::Unit::HOST);
    Memory<uint64_t>::copy(n_of_thread_cycles, Cuda_Utilities::Unit::DEVICE, n_of_thread_cycles, Cuda_Utilities::Unit::HOST);
    Memory<uint64_t>::copy(checksum,           Cuda_Utilities::Unit::DEVICE, checksum,           Cuda_Utilities::Unit::HOST);
	Memory<uint64_t>::copy(residues,           Cuda_Utilities::Unit::DEVICE, residues,           Cuda_Utilities::Unit::HOST);
	Memory<bool>::copy(skip_task,              Cuda_Utilities::Unit::DEVICE, skip_task,          Cuda_Utilities::Unit::HOST);

    while (true) {
        auto start_GPU = std::chrono::high_resolution_clock::now();

        collatz<<<n_of_survivors / n_of_threads + 1, n_of_threads, 0, 0>>>(start_point.get_device_address(), n_of_thread_cycles.get_device_address(),
                                                                           residues.get_device_address(),    checksum.get_device_address(),  
                                                                           skip_task.get_device_address());

        Cuda_Utilities::cuda_synchronize();

        auto                                      end_GPU      = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_GPU = end_GPU - start_GPU;
        auto                                      start_CPU    = std::chrono::high_resolution_clock::now();

        Memory<uint64_t>::copy(checksum, Cuda_Utilities::Unit::HOST, checksum,  Cuda_Utilities::Unit::DEVICE);
		Memory<bool>::copy(skip_task,    Cuda_Utilities::Unit::HOST, skip_task, Cuda_Utilities::Unit::DEVICE);

		const uint64_t temp_start    = start_point[0];
        const uint64_t temp_checksum = checksum[0];
        const bool     temp_skip     = skip_task[0];

        start_point[0] += n_of_thread_cycles[0];
        checksum[0]    =  0;
		skip_task[0]   =  false;

		Memory<uint64_t>::copy(start_point, Cuda_Utilities::Unit::DEVICE, start_point, Cuda_Utilities::Unit::HOST);
        Memory<uint64_t>::copy(checksum,    Cuda_Utilities::Unit::DEVICE, checksum,    Cuda_Utilities::Unit::HOST);
		Memory<bool>::copy(skip_task,       Cuda_Utilities::Unit::DEVICE, skip_task,   Cuda_Utilities::Unit::HOST);

        auto                                      end_CPU      = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_CPU = end_CPU - start_CPU;

        if (temp_skip)
            std::cout << "Task " << temp_start / n_of_thread_cycles[0] << " (" << temp_start << ") skipped because of 128bit overflow: check on CPU!\n";

		std::cout << "Task size:         2^" << collatz_sieve + (uint8_t)std::log2(n_of_thread_cycles[0]) << "\n";
        std::cout << "Current Task:      "   << temp_start / n_of_thread_cycles[0] << " (" << temp_start << ")\n";
        std::cout << "GPU Time taken:    "   << (uint32_t)duration_GPU.count() << "ms\n";
        std::cout << "CPU Time taken:    "   << (uint32_t)std::ceil(duration_CPU.count()) << "ms\n";
        std::cout << "Checks per second: "   << (temp_skip ? "" : "2^") << (temp_skip ? 0 : std::log2((1ull << collatz_sieve) * n_of_thread_cycles[0] * 1000.0 / duration_GPU.count())) << "\n";
        std::cout << "Checksum:          "   << (temp_skip ? 0 : temp_checksum) << "\n\n";
    }

}
