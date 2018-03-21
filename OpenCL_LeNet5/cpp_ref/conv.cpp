/* A Simple Conv_Layer implementation Just for test */
#include <stdio.h>
#include "pgm.h"

typedef float IMG_PIX;

typedef struct {
	int rows;
	int cols;
	IMG_PIX *data
}Mat;

void print_output(float *out, int rows, int cols)
{
	int cur_r, cur_c;
	for(cur_r = 0; cur_r < rows; cur_r++)
	{
		for(cur_c = 0; cur_c < cols; cur_c)
		{
			printf("%f ", out[cur_r * cols + cur_c]);
		}
		printf("\n");
	}
}


//Custom filter2D based on cv::filter2D
void myFilter2D(const Mat &input, Mat &out, const Mat &kernel, float bias)
{
	for(int r = 0; r < input.rows - kernel.rows + 1; r++)
	{
		for(int c = 0; c < input.cols - kernel.cols + 1; c++)
		{
			out.data[r*input.cols + c] = 0;
			for(int kr = 0; kr < kernel.rows; kr++)
			{
				for(int kc = 0; kc < kernel.cols; kc++)
				{
					out.data[r*input.cols + c] += 
							kernel.data[kr*kernel.cols+kc] * input.data[(r+kr)*input.cols + c + kc]
				}
			}
			out.data[r*input.cols + c] += bias;
		}
	}
}

int main(int argc, char **argv)
{
	char *imgPath;
	pgm_t inputImg, filtImg;

	if(argc == 1)
	{
		printf("Useage: %s <ImgPath>(pgm Format)", argv[0]);
		exit(-1);
	}
	imgPath = argv[1];

	readPGM(&inputImg, imgPath);
	Mat normImg;
	normImg.data = (IMG_PIX *)malloc(inputImg.width * inputImg.height*sizeof(IMG_PIX) );
	normImg.rows = inputImg.height;
	normImg.cols = inputImg.width;

	//Image Normalization
	for(int i = 0; i < inputImg.width * inputImg.height; i++)
	{
		normImg[i] = (IMG_PIX)inputImg[i]/255.0;
	}

	//Test Conv With Avg Kernel
	Mat kernel, output;
	const int kernel_size = 3;
	IMG_PIX avg_ker[kernel_size * kernel_size] = {0.0623, 0.0623, 0.0623, 0.0623, 0.0623, 0.0623, 0.0623, 0.0623};
	kernel.data = avg_ker;
	kernel.rows = kernel_size;
	kernel.cols = kernel_size;

	output.data = (IMG_PIX *)malloc(inputImg.width * inputImg.height * sizeof(IMG_PIX));
	output.rows = inputImg.height;
	output.cols = inputImg.width;

	float bias = 0.01;
	myFilter2D(normImg, output, kernel, bias);
	filtImg.width = output.cols;
	filtImg.height = output.rows;

	normalizeF2PGM(&filtImg, output.data);
	writePGM(&filtImg, "output.pgm");

	return 0;

}