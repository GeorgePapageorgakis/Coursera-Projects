/**
    Histogram_CPU.cpp
    Purpose: Histogram Equalization. The purpose of this program is to implement an 
    efficient histogramming equalization algorithm for an input image. Like the image
    convolution MP, the image is represented as RGB float values. We will convert that
    to GrayScale unsigned char values and compute the histogram. Based on the 
    histogram, you will compute a histogram equalization function which you will then
    apply to the original image to get the colour corrected image.
    
    For developing on their own system. The images are stored in PPM (P6) format, this 
    means that you can (if you want) create your own input images. The easiest way to 
    create image is via external tools. You can use tools such as bmptoppm.
    
    @author George Papageorgakis
    @version 1.0 03/2015
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

//p is the probability of a pixel to be in a histogram bin
float p(unsigned int x, int len){
    return (float) ((float)(x)/(float)(len));
}
//limiting a position to an area, merely moves the point to the nearest available value.
float clamp(float x, float min, float max){
    if (x < min)
        return min;
    else if (x > max)
        return max;
	return x;
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
	float * host_cdf;
	unsigned char * host_ucharImage;
	unsigned char * host_grayImage;
	unsigned int * host_histogram;
	const char * inputImageFile;
	
    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");
  
    //hostA = ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
	//hostB = ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
	hostInputImageData  = ( float * )wbImage_getData(inputImage);
	hostOutputImageData = ( float * )wbImage_getData(outputImage);
	
	host_cdf		= (float *) 		malloc (HISTOGRAM_LENGTH * 							sizeof(float));
	host_histogram 	= (unsigned int *)	malloc (HISTOGRAM_LENGTH * 							sizeof(unsigned int));
	host_ucharImage = (unsigned char *) malloc (imageWidth * imageHeight * imageChannels * 	sizeof(unsigned char));
	host_grayImage 	= (unsigned char *) malloc (imageWidth * imageHeight *					sizeof(unsigned char));
		
	wbLog(TRACE, "Dimensions of input image: ", imageWidth, "x ", imageHeight);
	wbLog(TRACE, "Total pixels of input image: ", imageWidth * imageHeight);
    wbLog(TRACE, "Histogram size is ", HISTOGRAM_LENGTH);

	i=0;
	//Cast the image from float to unsigned char
	while (i < imageWidth * imageHeight * imageChannels){
		host_ucharImage[i] = (unsigned char) (255 * hostInputImageData[i]);
		i++;
	}
	
	
	//Convert the image from RGB to GrayScale
	i=0;
	unsigned char r, g, b;
	while (i < imageWidth * imageHeight * imageChannels){
		r = host_ucharImage[i];
        g = host_ucharImage[i + 1];
        b = host_ucharImage[i + 2];
        host_grayImage[i/imageChannels] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
		i += imageChannels;
	}
	
	//Compute the histogram of grayImage
	i=0;
	while (i < HISTOGRAM_LENGTH){
		host_histogram[i] = 0;
		i++;
	}
	i=0;
	while (i < imageWidth * imageHeight){
		host_histogram[host_grayImage[i]] += 1;
		i++;
	}
	
	//Compute the Comulative Distribution Function of histogram
	host_cdf[0] = p(host_histogram[0], imageWidth * imageHeight);
	for (i = 1; i< 256; i++){
		host_cdf[i] = host_cdf[i - 1] + p(host_histogram[i], imageHeight*imageWidth);
	}
	
	//Apply the histogram equalization function
	for (i = 0; i< imageHeight*imageWidth*imageChannels; i++){
		host_ucharImage[i] = clamp(255*(host_cdf[host_ucharImage[i]] - host_cdf[0])/(1 - host_cdf[0]), 0, 255);
	}
	
	
	i=0;
	//Cast the image from unsigned char to float
	while (i < imageWidth * imageHeight * imageChannels){
		hostOutputImageData[i] = (float) (host_ucharImage[i]/255.0);
		i++;
	}
	

	wbTime_stop(Copy, "Copying data from the GPU");
    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
    wbSolution(args, outputImage);

	free(host_cdf);
	free(host_histogram);
	free(host_ucharImage);
	free(host_grayImage);
    free(hostInputImageData);
    free(hostOutputImageData);

    return 0;
}
