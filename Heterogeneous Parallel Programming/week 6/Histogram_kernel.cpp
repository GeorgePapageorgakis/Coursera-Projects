/**
    MP7
    Histogram_kernel.cpp
    Purpose: Histogram Equalization CUDA API. Purpose of this program is to implement an 
    efficient histogramming equalization algorithm for an input image. Like the image 
    convolution MP, the image is represented as RGB float values. We will convert 
    that to GrayScale unsigned char values and compute the histogram. Based on the 
    histogram, you will compute a histogram equalization function which you will 
    then apply to the original image to get the colour corrected image.
    
    For developing on their own system. The images are stored in PPM (P6) format, 
    this means that you can (if you want) create your own input images. The easiest 
    way to create image is via external tools. You can use tools such as bmptoppm.
    
    Needs optimization for different sizes of histogram length and kernel block length.
    
    @author George Papageorgakis
    @version 1.0 04/2015
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
#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256

//p is the probability of a pixel to be in a histogram bin
__device__ float p(unsigned int x, int len){
    return (float) ((float)(x)/(float)(len));
}

//limiting a position to an area, merely moves the point to the nearest available value.
__device__ float clamp(float x, float min, float max){
    if (x < min)
        return min;
    else if (x > max)
        return max;
	return x;
}

//Cast the image from float to unsigned char
__global__ void castImageToUCharKernel(int len, float * deviceInputImageData, unsigned char * ucharImage){
	unsigned int tid	= blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x*gridDim.x;

	while (tid < len){
		ucharImage[tid] = (unsigned char) (255 * deviceInputImageData[tid]);
		__syncthreads();
		tid += stride;
	}
}

//Convert the image from RGB to GrayScale
__global__ void rgbToGrayScaleKernel(int len, unsigned char * grayImage, unsigned char * ucharImage){
	unsigned char r, g, b;
	unsigned int tid 	= blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x*gridDim.x;
	
	while (tid < len){
		r = ucharImage[3*tid];
        g = ucharImage[3*tid + 1];
        b = ucharImage[3*tid + 2];
        grayImage[tid] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
		__syncthreads();
		tid += stride;
	}
}

//Compute the histogram of grayImage
__global__ void histogramKernel(int len, unsigned char *grayImage, unsigned int *histogram){
	__shared__ unsigned int private_histo[HISTOGRAM_LENGTH];	
	unsigned int tid 	= blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x*gridDim.x;
	unsigned int t	 	= threadIdx.x;
	
	while (t < HISTOGRAM_LENGTH) {
		private_histo[t] = 0;
		t += blockDim.x;
	}
	__syncthreads();
	//Build Private Histogram
	while (tid < len) {
		atomicAdd(&(private_histo[grayImage[tid]]), 1);
		tid += stride;
	}
	__syncthreads();
	t = threadIdx.x;
	//Build Final Histogram 
	while (t < HISTOGRAM_LENGTH) {
		atomicAdd(&(histogram[t]), private_histo[t]);
		t += blockDim.x;
	}
}

//Computes the Comulative Distribution Function of histogram (scan operation)
__global__ void cdfHistoKernel(int len, float *cdf, unsigned int *histogram){
	__shared__ float buffer [BLOCK_SIZE];
	
	//Load a segment of the input vector into shared memory
	buffer[threadIdx.x] = p(histogram[threadIdx.x], len);
	__syncthreads();
	// BLOCK_SIZE is in shared memory
	for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
		unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
		if(index < blockDim.x)
			buffer[index] += buffer[index - stride];
		__syncthreads();
	}
	//Post Reduction Reverse Phase Kernel Code
	for (unsigned int stride = blockDim.x / 4; stride > 0;	stride >>= 1) {
		__syncthreads();
		unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
		if(index + stride < blockDim.x) {
			buffer[index + stride] += buffer[index];
		}
	}
	__syncthreads();
	//copy results to the output array in global memory
	cdf[threadIdx.x] = buffer[threadIdx.x];
	__syncthreads();
}

//The histogram equalization function (correct) remaps the 
//cdf of the histogram of the image to a linear function
//Apply the histogram equalization function
__global__ void remapCDFKernel(	int len, float * cdf, unsigned char * ucharImage){
	unsigned int tid 	= blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x*gridDim.x;
	float cdfmin = cdf[0];
	
	while (tid < len){
		ucharImage[tid] = clamp(255*(cdf[ucharImage[tid]] - cdfmin)/(1 - cdfmin), 0, 255);
		tid += stride;
		__syncthreads();
	}
}

//Cast the image from unsigned char to float
__global__ void castImageToFloatKernel(int len, float * deviceOutputImageData, unsigned char * ucharImage){
	unsigned int tid	= blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x*gridDim.x;

	while (tid < len){
		deviceOutputImageData[tid] = (float) (ucharImage[tid]/255.0);
		__syncthreads();
		tid += stride;
	}
}

int main(int argc, char ** argv) {
    wbArg_t args;
	int i;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * deviceInputImageData;
	float * deviceOutputImageData;
	float * cdf;
	unsigned char * ucharImage;
	unsigned char * grayImage;
	unsigned int * histogram;
	const char * inputImageFile;
	dim3 dimBlock, dimGrid;
	
    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage 		= wbImport(inputImageFile);
    imageWidth 		= wbImage_getWidth(inputImage);
    imageHeight 	= wbImage_getHeight(inputImage);
    imageChannels 	= wbImage_getChannels(inputImage);
    outputImage 	= wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");
  
	hostInputImageData  = ( float * )wbImage_getData(inputImage);
	hostOutputImageData = ( float * )wbImage_getData(outputImage);
		
	wbTime_start(GPU, "Doing GPU memory allocation");
    wbCheck(cudaMalloc((void **) &deviceInputImageData,  imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
	wbCheck(cudaMalloc((void **) &ucharImage, 			 imageWidth * imageHeight * imageChannels * sizeof(unsigned char)));
	wbCheck(cudaMalloc((void **) &grayImage, 			 imageWidth * imageHeight * 				sizeof(unsigned char)));
	wbCheck(cudaMalloc((void **) &histogram,  			 HISTOGRAM_LENGTH *							sizeof(unsigned int)));
	wbCheck(cudaMalloc((void **) &cdf,  			 	 HISTOGRAM_LENGTH *							sizeof(float)));
    wbTime_stop(GPU, "Doing GPU memory allocation");
		
	wbLog(TRACE, "Dimensions of input image: ", imageWidth, "x ", imageHeight);
	wbLog(TRACE, "Total pixels of input image: ", imageWidth * imageHeight);
    wbLog(TRACE, "Histogram size is ", HISTOGRAM_LENGTH);

	
	wbTime_start(Copy, "Copying data to the GPU");
    wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(Copy, "Copying data to the GPU");
	
	
	wbTime_start(Compute, "Doing the computation on the GPU...");
	
	//Cast the image from float to unsigned char
	dimBlock = dim3 (BLOCK_SIZE, 1, 1);
	dimGrid  = dim3 ((imageWidth*imageHeight*imageChannels - 1)/BLOCK_SIZE + 1, 1, 1);
	wbLog(TRACE, "Casting Input Image to Unsigned char...");
	castImageToUCharKernel <<<dimGrid, dimBlock>>> (imageHeight*imageWidth*imageChannels, deviceInputImageData, ucharImage);

	//Convert the image from RGB to GrayScale
	dimGrid = dim3((imageWidth*imageHeight - 1)/BLOCK_SIZE + 1, 1, 1);
	wbLog(TRACE, "Converting Input Image to Gray Scale...");
	rgbToGrayScaleKernel <<<dimGrid, dimBlock>>> (imageHeight*imageWidth, grayImage, ucharImage);
	
	//Compute the histogram of grayImage
	wbLog(TRACE, "Calculating Histogram of Gray Scale image...");
	histogramKernel <<<dimGrid, dimBlock>>> (imageHeight*imageWidth, grayImage, histogram);
	
	//Compute the Comulative Distribution Function of histogram
	wbLog(TRACE, "Calculating CDF of the Histogram...");
	dimGrid = dim3((HISTOGRAM_LENGTH-1)/(BLOCK_SIZE) + 1, 1, 1);
	cdfHistoKernel <<<dimGrid, dimBlock>>> (imageWidth*imageHeight, cdf, histogram);
	
	//Apply the histogram equalization function
	wbLog(TRACE, "Applying the histogram equalization function...");
	dimGrid  = dim3 ((imageWidth*imageHeight*imageChannels - 1)/BLOCK_SIZE + 1, 1, 1);
	remapCDFKernel <<<dimGrid, dimBlock>>> (imageHeight*imageWidth*imageChannels, cdf, ucharImage);
		
	//Cast the image from unsigned char to float
	wbLog(TRACE, "Casting Input Image back to float...");
	castImageToFloatKernel	<<<dimGrid, dimBlock>>>	(imageHeight*imageWidth*imageChannels, deviceOutputImageData, ucharImage);
	
	wbTime_stop(Compute, "Doing the computation on the GPU");
	
	
    wbTime_start(Copy, "Copying data from the GPU");
    wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	wbTime_stop(Copy, "Copying data from the GPU");
   
	wbSolution(args, outputImage);

	
	cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
	cudaFree(cdf);
	cudaFree(ucharImage);
	cudaFree(grayImage);
	cudaFree(histogram);

    free(hostInputImageData);
    free(hostOutputImageData);

    return 0;
}
