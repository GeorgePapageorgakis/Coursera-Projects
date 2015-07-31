/**
    vector_addition_streams_cuda.cpp
    Purpose: A simple vector Addition using streams on CUDA API
    
    @author George Papageorgakis
    @version 1.0 04/2015
*/
#include	<wb.h>

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
	//vector addition
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
		out[i] = in1[i] + in2[i];
}


int main(int argc, char ** argv) {
    	wbArg_t args;
    	int inputLength;
	int SegSize = 64;
    	float * hostInput1;
    	float * hostInput2;
    	float * hostOutput;
    	float * deviceInput1;
    	float * deviceInput2;
	float *d_A0, *d_B0, *d_C0;// device memory for stream 0
	float *d_A1, *d_B1, *d_C1;// device memory for stream 1
	float *d_A2, *d_B2, *d_C2;// device memory for stream 2
	float *d_A3, *d_B3, *d_C3;// device memory for stream 3
	float *h_A, *h_B, *h_C;// host memory for stream 0
    	float * deviceOutput;
	//Create streams
	cudaStream_t stream0, stream1, stream2, stream3;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	
    	args = wbArg_read(argc, argv);

    	wbTime_start(Generic, "Importing data and creating memory on host");
    	hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    	hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
	hostOutput = (float *) malloc(inputLength * sizeof(float));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	//@@ Allocate GPU memory here
	cudaMalloc((void **) &deviceInput1, inputLength * sizeof(float));
	cudaMalloc((void **) &deviceInput2, inputLength * sizeof(float));
	cudaMalloc((void **) &deviceOutput, inputLength * sizeof(float));
	
	//assign all stream array pointers to the main arrays
	d_A0 = deviceInput1;
	d_A1 = deviceInput1;
	d_A2 = deviceInput1;
	d_A3 = deviceInput1;
	
	d_B0 = deviceInput2;
	d_B1 = deviceInput2;
	d_B2 = deviceInput2;
	d_B3 = deviceInput2;
	
	d_C0 = deviceOutput;
	d_C1 = deviceOutput;
	d_C2 = deviceOutput;
	d_C3 = deviceOutput;
	
	h_A = hostInput1;
	h_B = hostInput2;
	h_C = hostOutput;
	
	wbTime_start(Compute, "Performing CUDA computation");
	//A Better Multi-Stream Host Code
	for (int i = 0; i < inputLength; i += SegSize * 4) {
		cudaMemcpyAsync(d_A0, h_A+i, 			SegSize*sizeof(float),cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_B0, h_B+i, 			SegSize*sizeof(float),cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_A1, h_A+i+SegSize, 	SegSize*sizeof(float),cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(d_B1, h_B+i+SegSize, 	SegSize*sizeof(float),cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(d_A2, h_A+i+(2*SegSize),SegSize*sizeof(float),cudaMemcpyHostToDevice, stream2);
		cudaMemcpyAsync(d_B2, h_B+i+(2*SegSize),SegSize*sizeof(float),cudaMemcpyHostToDevice, stream2);
		cudaMemcpyAsync(d_A3, h_A+i+(3*SegSize),SegSize*sizeof(float),cudaMemcpyHostToDevice, stream3);
		cudaMemcpyAsync(d_B3, h_B+i+(3*SegSize),SegSize*sizeof(float),cudaMemcpyHostToDevice, stream3);
		
		vecAdd<<<SegSize/256, 256, 0, stream0>>>(d_A0, d_B0, d_C0, SegSize);
		vecAdd<<<SegSize/256, 256, 0, stream1>>>(d_A1, d_B1, d_C1, SegSize);
		vecAdd<<<SegSize/256, 256, 0, stream2>>>(d_A2, d_B2, d_C2, SegSize);
		vecAdd<<<SegSize/256, 256, 0, stream3>>>(d_A3, d_B3, d_C3, SegSize);
		
		cudaStreamSynchronize(stream0);
		cudaMemcpyAsync(h_C+i, 			d_C0, SegSize*sizeof(float),cudaMemcpyDeviceToHost, stream0);
		cudaStreamSynchronize(stream1);
		cudaMemcpyAsync(h_C+i+SegSize, 	d_C1, SegSize*sizeof(float),cudaMemcpyDeviceToHost, stream1);
		cudaStreamSynchronize(stream2);
		cudaMemcpyAsync(h_C+i+2*SegSize,d_C2, SegSize*sizeof(float),cudaMemcpyDeviceToHost, stream2);
		cudaStreamSynchronize(stream3);
		cudaMemcpyAsync(h_C+i+3*SegSize,d_C3, SegSize*sizeof(float),cudaMemcpyDeviceToHost, stream3);
	}
	
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");
	
	
    	wbSolution(args, hostOutput, inputLength);
	
	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream3);
	cudaFree(deviceInput1);
	cudaFree(deviceInput2);
	cudaFree(deviceOutput);
    	free(hostInput1);
    	free(hostInput2);
    	free(hostOutput);
    	return 0;
}
