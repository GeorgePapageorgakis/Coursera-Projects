/**
    MP4
    cuda_convolution_2D_tiled.cpp
    Purpose: Implement a tiled image convolution using both shared and constant memory CUDA API
    We will have a constant 5x5 convolution mask, but will have arbitrarily sized image 
    (We will assume the image dimensions are greater than 5x5 in this Lab).
    
    @author George Papageorgakis
    @version 1.0 02/2015
*/
#include    <wb.h>
#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define MASK_WIDTH  5
#define MASK_RADIUS MASK_WIDTH/2
#define O_TILE_WIDTH 12
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH -1)


__global__ void convolution_2D_kernel(float *P, float *N, int height, int width, int channels, const float * __restrict__ M) {
	__shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];
	
	//total dimensions of the picture are: O_TILE_WIDTH x O_TILE_WIDTH*channels
	//Shifting from output coordinates to input coordinate
	int row_o = blockIdx.y*O_TILE_WIDTH + threadIdx.y;
	int col_o = blockIdx.x*(O_TILE_WIDTH*channels) + threadIdx.x*channels;
	int row_i = row_o - MASK_RADIUS;
	int col_i = col_o - MASK_RADIUS*channels;
	
	//fetch from GLOBAL memory to SHARED memory, coalesced memory access	
	//accessing image3D[i][j][z] is same as image3D[ i*cols+j + rows*cols*z];
	
	//Taking Care of Boundaries
	for (int color=0; color < channels; color++){
		//All Threads Participate in Loading Input Tiles
		//Load Halo or Ghost elements
		if((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width*channels) ) {
			Ns[threadIdx.y][threadIdx.x] = N[row_i*width*channels + col_i + color];
		} 
		else{
			Ns[threadIdx.y][threadIdx.x] = 0.0f;
		}
		__syncthreads();
		
		float output = 0.0f;
		//Threads within the O_TILE_WIDTH participate in convolution calculation
		//Size of threadblock is actually bigger than the output tile, (halo elements)
		//We are only using the first threads within the range of O_TILE_WIDTH to 
		//calculate the output tile and the rest of the threads will be set to idle 
		if(threadIdx.y < O_TILE_WIDTH && threadIdx.x < O_TILE_WIDTH){
			for(int row = 0; row < MASK_WIDTH; row++) {
				for(int col = 0; col < MASK_WIDTH; col++) {
					output += M[row * MASK_WIDTH + col] * Ns[row+threadIdx.y][col+threadIdx.x];
				}
			}
		}
		__syncthreads();
		
		//Write back the calculated value in the corresponding pixel
		if(threadIdx.y < O_TILE_WIDTH && threadIdx.x < O_TILE_WIDTH && row_o < height && col_o < width*channels)
			P[row_o*width*channels + col_o + color] = output;
		__syncthreads();
	}
}	

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData, hostMaskData, maskRows * maskColumns * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	dim3 dimGrid((imageWidth-1)/O_TILE_WIDTH + 1, (imageHeight-1)/O_TILE_WIDTH + 1, 1);
    convolution_2D_kernel<<<dimGrid, dimBlock>>>(	deviceOutputImageData, 
										deviceInputImageData,
										imageHeight, 
										imageWidth, 
										imageChannels, 
										deviceMaskData);
	wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
