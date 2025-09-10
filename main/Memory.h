#pragma once

// Inclusions
#include <cstdint>
#include "cuda_runtime.h"
#include "Cuda_Utilities.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <memory>
#include <string>
#include <utility>

template<typename T> class Memory {

public:

	// Constructors
	Memory(const uint64_t& _n_of_elements,
		   const bool& _allocate_on_host = true,
		   const bool& _allocate_on_device = true) : n_of_elements(_n_of_elements),
													 host_memory(_allocate_on_host ? sizeof(T) * _n_of_elements : 0),
													 device_memory(_allocate_on_device ? sizeof(T) * _n_of_elements : 0) {

		if (_allocate_on_host)
			allocate_host(_n_of_elements);
		
		if (_allocate_on_device) 
			allocate_device(_n_of_elements);
		
	}
	Memory(const Memory& _other) : n_of_elements(_other.n_of_elements),
								   host_memory(_other.host_memory),
								   device_memory(_other.device_memory) {

		if (_other.local_address) {
			local_address = std::make_unique<T[]>(_other.n_of_elements);

			std::copy(_other.local_address.get(), _other.local_address.get() + _other.n_of_elements, local_address.get());

			Cuda_Utilities::increase_used_host_memory(host_memory);
		}

		if (_other.device_address) {
			cudaError_t cudaStatus1 = cudaMalloc((void**)&device_address, _other.n_of_elements * sizeof(T));

			if (cudaStatus1 != cudaSuccess) {
				std::cerr << "cudaMalloc FAILED with error: " << cudaGetErrorString(cudaStatus1) << std::endl;
				return;
			}

			Cuda_Utilities::increase_used_device_memory(device_memory);

			cudaError_t cudaStatus2 = cudaMemcpy(device_address, _other.device_address, device_memory, cudaMemcpyDeviceToDevice);

			if (cudaStatus2 != cudaSuccess) {
				std::cerr << "cudaMalloc FAILED with error: " << cudaGetErrorString(cudaStatus2) << std::endl;
				return;
			}

		}

	}
	Memory(Memory&& _other) noexcept : local_address(std::move(_other.local_address)),
									   device_address(std::move(_other.device_address)),
									   n_of_elements(std::move(_other.n_of_elements)),
									   host_memory(std::move(_other.host_memory)),
									   device_memory(std::move(_other.device_memory)) {
		
        _other.local_address  = nullptr;
		_other.device_address = nullptr;
		_other.n_of_elements  = 0;
		_other.host_memory      = 0;
		_other.device_memory    = 0;
	}

	// Destructor
	~Memory() {
		deallocate_host();

		if (Cuda_Utilities::get_device_status())
			deallocate_device();

		n_of_elements = 0;
		host_memory     = 0;
		device_memory   = 0;
	}

	// Copy assignment operator
	Memory& operator=(const Memory& _other) {

		if (this != &_other) {
			deallocate_host();

			if (_other.local_address != nullptr) {
				allocate_host(_other.n_of_elements);
				std::copy(_other.local_address.get(), _other.local_address.get() + _other.n_of_elements, local_address.get());
			}

			deallocate_device();

			if (_other.device_address != nullptr) {
				allocate_device(_other.n_of_elements);
				copy(*this, Cuda_Utilities::Unit::DEVICE, _other, Cuda_Utilities::Unit::DEVICE, true);
			}

		}

		return *this;
	}

	// Move assignment operator
	Memory& operator=(Memory&& _other) noexcept {

		if (this != &_other) {
			deallocate_host();
			deallocate_device();

			local_address  = std::move(_other.local_address);
			device_address = std::move(_other.device_address);
			n_of_elements  = std::move(_other.n_of_elements);
			host_memory      = std::move(_other.host_memory);
			device_memory    = std::move(_other.device_memory);

			_other.local_address  = nullptr;
			_other.device_address = nullptr;
			_other.n_of_elements  = 0;
			_other.host_memory      = 0;
			_other.device_memory    = 0;
		}

		return *this;
	}

	// Subscript operator
	T& operator[](const uint64_t& _index) {

		if (!local_address) {
			std::cerr << "Memory not allocated on host" << std::endl;
			std::exit(-1);
		}

		if (_index >= n_of_elements) {
			std::cerr << "Index out of bounds: " << _index << " >= " << n_of_elements << std::endl;
			std::exit(-1);
		}

		return local_address[_index];
	}

	// Getters
	T*     get_host_address() {
		return local_address.get();
	}
	T*     get_device_address() {
		return device_address;
	}
	uint64_t get_n_of_elements() const {
		return n_of_elements;
	}
	uint64_t get_used_host_memory() const {
		return host_memory;
	}
	uint64_t get_used_device_memory() const {
		return device_memory;
	}

	// Methods
	std::string to_string() {
		std::string result = "Memory:\n";

		result += "  Number of elements: " + std::to_string(n_of_elements)                                    + "\n";
		result += "  Host size:          " + std::to_string(host_memory)                                        + " bytes\n";
		result += "  Device size:        " + std::to_string(device_memory)                                      + " bytes\n";
		result += "  Host address:       " + std::to_string(reinterpret_cast<uintptr_t>(local_address.get())) + "\n";
		result += "  Device address:     " + std::to_string(reinterpret_cast<uintptr_t>(device_address))      + "\n";

		return result;
	}

	// Static methods
	static void copy(Memory<T>& _dst,
		             const Cuda_Utilities::Unit& _dst_unit,
		             const Memory<T>& _src,
					 const Cuda_Utilities::Unit& _src_unit,
		             const bool& _wait = true) {

		if (&_dst == &_src && _dst_unit == _src_unit) {
			std::cerr << "Source and destination memory are the same, no need to copy" << std::endl;
			return;
		}

		if (_dst.n_of_elements != _src.n_of_elements) {
			std::cerr << "Memory number of elements do not match: " << _dst.n_of_elements << " != " << _src.n_of_elements << std::endl;
			return;
		}

		if (_dst_unit == Cuda_Utilities::Unit::HOST && _src_unit == Cuda_Utilities::Unit::HOST)
			std::copy(_src.local_address.get(), _src.local_address.get() + _src.n_of_elements, _dst.local_address.get());

		else if (_dst_unit == Cuda_Utilities::Unit::DEVICE && _src_unit == Cuda_Utilities::Unit::DEVICE) {
			cudaError_t cudaStatus = cudaMemcpyAsync(_dst.device_address, _src.device_address, _src.device_memory, cudaMemcpyDeviceToDevice);

			if (cudaStatus != cudaSuccess) {
				std::cerr << "cudaMemcpy FAILED with error: " << cudaGetErrorString(cudaStatus) << std::endl;
				return;
			}

		}

		else if (_dst_unit == Cuda_Utilities::Unit::HOST && _src_unit == Cuda_Utilities::Unit::DEVICE) {
			cudaError_t cudaStatus = cudaMemcpyAsync(_dst.local_address.get(), _src.device_address, _src.device_memory, cudaMemcpyDeviceToHost);

			if (cudaStatus != cudaSuccess) {
				std::cerr << "cudaMemcpy FAILED with error: " << cudaGetErrorString(cudaStatus) << std::endl;
				return;
			}

		}

		else if (_dst_unit == Cuda_Utilities::Unit::DEVICE && _src_unit == Cuda_Utilities::Unit::HOST) {
			cudaError_t cudaStatus = cudaMemcpyAsync(_dst.device_address, _src.local_address.get(), _src.host_memory, cudaMemcpyHostToDevice);

			if (cudaStatus != cudaSuccess) {
				std::cerr << "cudaMemcpy FAILED with error: " << cudaGetErrorString(cudaStatus) << std::endl;
				return;
			}

		}

		if (_wait)
			Cuda_Utilities::cuda_synchronize();

	}
	
	// Methods
	void allocate_host(const uint64_t& _n_of_elements) {

		if (local_address == nullptr) {
			local_address = std::make_unique<T[]>(_n_of_elements);
			n_of_elements = _n_of_elements;
			host_memory = sizeof(T) * _n_of_elements;
			Cuda_Utilities::increase_used_host_memory(host_memory);
		}

		else {
			std::cerr << "Host memory already allocated" << std::endl;
			return;
		}

	}
	void allocate_device(const uint64_t& _n_of_elements) {

		if (device_address == nullptr) {
			cudaError_t cudaStatus = cudaMalloc((void**)&device_address, _n_of_elements * sizeof(T));

			if (cudaStatus != cudaSuccess) {
				std::cerr << "cudaMalloc FAILED with error: " << cudaGetErrorString(cudaStatus) << std::endl;
				return;
			}

			n_of_elements = _n_of_elements;
			device_memory = sizeof(T) * _n_of_elements;
			Cuda_Utilities::increase_used_device_memory(device_memory);
		}

		else {
			std::cerr << "Device memory already allocated" << std::endl;
			return;
		}

	}
	void deallocate_host() {

		if (local_address != nullptr) {
			local_address.reset();
			Cuda_Utilities::decrease_used_host_memory(host_memory);
			host_memory = 0;
		}

	}
	void deallocate_device() {

		if (device_address != nullptr) {
			cudaError_t cudaStatus = cudaFree(device_address);

			if (cudaStatus != cudaSuccess) {
				std::cerr << "cudaFree FAILED with error: " << cudaGetErrorString(cudaStatus) << std::endl;
				return;
			}

			device_address = nullptr;
			Cuda_Utilities::decrease_used_device_memory(device_memory);
			device_memory = 0;
		}

	}

private:

	// Fields
	std::unique_ptr<T[]> local_address  = nullptr; // Host memory address
	T*					 device_address = nullptr; // Device memory address
	uint64_t               n_of_elements  = 0;       // Number of elements allocated
	uint64_t               host_memory    = 0;       // Allocated host memory in bytes   (Implicitly telling if it has been host-allocated)
	uint64_t               device_memory  = 0;       // Allocated device memory in bytes (Implicitly telling if it has been device-allocated)

};