/**
    MP6
    list_scan.cpp
    Purpose: MP Scan. Given a list (lst) of length n Output its prefix sum = {lst[0], lst[0] + lst[1],
    lst[0] + lst[1] + ... + lst[n-1]}
    Implement a kernel that performs parallel scan on a 1D list. The scan operator used will be addition.
    A work efficient kernel should be able to handle input lists of arbitrary length. However, for 
    simplicity, you can assume that the input list will be at most 2048 * 65,535 elements so that it can
    be handled by only one kernel launch.The boundary condition can be handled by filling “identity value
    (0 for sum)” into the shared memory of the last block when the length is not a multiple of the 
    thread block size.
    
    @author George Papageorgakis
    @version 1.0 03/2015
*/
#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)
    
__global__ void scan_kernel(float * input, float * output, int inlen) {
	__shared__ float XY[2*BLOCK_SIZE];
	unsigned int tid 			= 2*blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int global_stride 	= 2*blockDim.x * gridDim.x;
	//TODO skepsou an to inlen einai mikrotero ti ginete... bug!
	if (inlen % BLOCK_SIZE != 0){
		XY[threadIdx.x			   ] = 0;
		XY[threadIdx.x + blockDim.x] = 0;
	}
	__syncthreads();
	
	while (tid < inlen){
		//Load a segment of the input vector into shared memory
		XY[threadIdx.x			   ] = input[tid];
		XY[threadIdx.x + blockDim.x] = input[tid + blockDim.x];
		__syncthreads();
		
		// 2*BLOCK_SIZE is in shared memory
		for (int stride = 1; stride <= BLOCK_SIZE; stride <<= 1) {
			int index = (threadIdx.x+1)*stride*2 - 1;
			if(index < 2*BLOCK_SIZE)
				XY[index] += XY[index-stride];
			__syncthreads();
		}
		
		//Post Reduction Reverse Phase Kernel Code
		for (int stride = BLOCK_SIZE/2; stride > 0;	stride >>= 1) {
			__syncthreads();
			int index = (threadIdx.x+1)*stride*2 - 1;
			if(index+stride < 2*BLOCK_SIZE) {
				XY[index + stride] += XY[index];
			}
		}
		__syncthreads();
		//copy results to the output array in global memory
		if (tid < inlen) 
			output[tid] = XY[threadIdx.x];
		if (tid + blockDim.x < inlen) 
			output[tid + blockDim.x] = XY[threadIdx.x + blockDim.x];		
		__syncthreads();
		tid += global_stride;
		if (inlen % BLOCK_SIZE != 0){
			XY[threadIdx.x] = 0;
			XY[threadIdx.x + blockDim.x] = 0;
		}
		__syncthreads();
	}
}

__global__ void scan_batches_kernel(float * output, int inlen) {
	__shared__ float limitVal;
	unsigned int tid = 2*blockDim.x + threadIdx.x;
	unsigned int global_stride = 2*blockDim.x;
	
	while (tid < inlen){
		//load the last value of the previous batch
		if (threadIdx.x == 0){
			if (tid - 1 > 0)
				limitVal = output[tid - 1];
			else
				limitVal = 0;
		}
		__syncthreads();
		//add the value to each element in the output array
		output[tid] += limitVal;
		if (tid + blockDim.x < inlen) 
			output[tid + blockDim.x] += limitVal;
		__syncthreads();
		tid += global_stride;
	}
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((numElements-1)/(2*BLOCK_SIZE) + 1, 1, 1);
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
	scan_kernel<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements);
	scan_batches_kernel<<<1, dimBlock>>>(deviceOutput, numElements);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}

