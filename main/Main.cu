
// Inclusions
#include "cuda_runtime.h"
#include "Cuda_Utilities.h"
#include "device_launch_parameters.h"
#include "Kernels.cuh"

int main() {
	Cuda_Utilities::cuda_choose_device();
	Kernels::launch_collatz();
	Cuda_Utilities::cuda_device_reset();

    return 0;
}