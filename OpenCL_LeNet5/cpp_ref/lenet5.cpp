/*
 * A simple Implementation of LeNet-5
 */

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <vector>
#include "lenet5.h"

using namespace cv;
using namespace std;

void loadConvParams(ConvParams &params, const float *w, const float *b)
{
	for(int n; n < params.outputNums; n++ )
	{
		vector<Mat> filt3D;
		for(int m; m < params.inputNums; m++)
		{
			Mat filt2D = Mat(params.filterH, params.filterW, CV_32F);
			for(int r = 0; r<params.filterH; r++)
			{
				for(int c = 0; c < params.filterW; c++)
				{
					filt2D.at<float>(r, c) = w[n*params.inputNums*params.filterH*filterW
						+m*params.filterH*params.filterW + r*params.filterH + c];
				}
			}
			filt3D.push_back(filt2D.clone());
		}
		params.W.push_back(filt3D);
	}

	params.bias = vector<float>(params.outputNums, 0);
	for(int n=0; n<params.outputNums; n++)
	{
		params.bias[n] = b[n];
	}
}

void loadFCParams(FCParams &params, const float *w, const float *b)
{
	params.W = Mat(params.outputNums, params.inputNums, CV_32F);
	params.bias = vector<float>(params.outputNums, 0);
	for(int r = 0; r< params.outputNums; r++)
	{
		for(int c =0; c < params.outputNums; c++)
		{
			params.W.at<float>(r, c) = w[r*params.inputNums+c];
		}
		params.bias[r] = b[r];
	}
}

void intiLeNet5Model(ConvLayers &conv, FCLayers &fc)
{
	conv.layersNum = 2;
	ConvParams convParam;

	//Load Conv Layer 1
	convParam.filterH = CONV1_FILTER_HEIGHT;
	convParam.filterW = CONV1_FILTER_WIDTH;
	convParam.inputNums = CONV1_INPUTS_NUMS;
	convParam.outputNums = CONV1_OUTPUTS_NUMS;
	loadConvParams(convParam, (const float*)conv1_weights, (const float*)conv1_bias);
	conv.layerParams.push_back(convParam);

	convParam.W.clear();
	convParam.bias.clear();

	//Load Conv Layer2
	convParam.filterH = CONV2_FILTER_HEIGHT;
	convParam.filterW = CONV2_FILTER_WIDTH;
	convParam.inputNums = CONV2_INPUTS_NUMS;
	convParam.outputNums = CONV2_OUTPUTS_NUMS;
	loadConvParams(convParam, (const float*)conv2_weights, (const float*)conv2_bias);
	conv.layerParams.push_back(convParam);
	
	fc.layersNum = 2;
	FCParams fcParam1, fcParam2;
	fcParam1.inputNums = FC1_INPUTS_NUMS;
	fcParam1.outputNums = FC1_OUTPUTS_NUMS;
	loadFCParams(fcParam1, (const float*)fc1_weights, (const float*)fc1_bias);
	fc.layerParams.push_back(fcParam1);

	fcParam2.inputNums = FC2_INPUTS_NUMS;
	fcParam2.outputNums = FC2_OUTPUTS_NUMS;
	loadFCParams(fcParam2, (const float*)fc2_weights, (const float*)fc2_bias);
	fc.layerParams.push_back(fcParam2);
}

//Layers Define
void customFilter2D(const Mat &input, Mat &out, const Mat &kernel)
{
	for (int row = 0; row < input.rows - kernel.rows + 1; row++)
	{
		for (int col = 0; col < input.col - kernel.col + 1; col++)
		{
			out.at<float>(row, col) = 0;
			for (int ker_r; ker_r < kernel.rows; ker_r++)
			{
				for (int ker_c = 0; ker_c < kernel.cols; ker_c)
				{
					out.at<float>(row, col) += kernel.at<float>(ker_r, ker_c) * input.at <float>(row + ker_r, col + ker_c);
				}
			}
		}
	}
}
//Map Accumulator
void mapAdd(const Mat &map, Mat &sum)
{
	assert(map.size() == sum.size());
	for (int row = 0; row < map.rows; row++)
	{
		for (int col = 0; col < map.cols; col++)
		{
			sum.at<float>(row, col) += map.at<float>(row, col);
		}
	}
}

//Convolution assuming no padding
void convLayer(const vector<Mat> &input, vector<Mat> &output, const ConvParams &params)
{
	Mat filter = Mat(input[0].rows - params.filterH + 1, input[0].cols - params.filterW + 1, CV_32F);
	for (int n = 0; n < params.outputNums; n++)
	{
		Mat out = Mat::zeros(input[0].rows - params.filterH + 1, input[0].cols - params.filterW + 1, CV_32F);
		for (int m = 0; m < params.outputNums; m++)
		{
			customFilter2D(input[m], filter, params.W[n][m]);
		}

		for (int r = 0; r < out.rows; r++)
		{
			for (int c = 0; c < out.cols; c++)
			{
				out.at<float>(r, c) += params.bias[n];
			}
		}
		output.push_back(out.clone());
	}
}

//Max Pooling Layer
void maxPoolLayer(const vector<Mat> &input, vector<Mat> &output, int winsize, int stride)
{
	int outH, outW;
	outH = (input[0].rows - winsize) / stride + 1;
	outW = (input[0].cols - winsize) / stride + 1;

	Mat out = Mat::zeros(outH, outW, CV_32F);

	for (int n = 0; n < input.size(); n++)
	{
		for (int r= 0; r < input[n].rows; r += stride)
		{
			for (int c = 0; c < input[n].cols; c += stride)
			{
				Mat temp;
				Point minP, maxP;
				double minVal, maxVal;
				Rect roi = Rect(c, r, winsize, winsize);
				input[n](roi).copyTo(temp);
				minMaxLoc(temp, &minVal, &maxVal);

				out.at<float>(r / stride, c / stride) = (float)maxVal;
			}
		}
		output.push_back(out.clone());
	}
}

void reluLayer(vector<float> &input)
{
	for (int i; i < input.size(); i++)
	{
		if (input[i] < 0.0)
			input[i] = 0.0;
	}
}

//Fully Connected Layer(Inner Product)
void fcLayer(const vector<float> &input, vector<float> &output, const FCParams &params)
{
	float sum;
	for (int n = 0; n < params.outputNums; n++)
	{
		sum = 0;
		for (int m = 0; m < params.inputNums; m++)
		{
			sum += input[m] * params.W.at<float>(n, m);
		}
		sum += params.bias[n];
		output.push_back(sum);
	}
}

//Fully Connected Layer input is a set of feature map(Inner Product)
void fcLayer(const vector<Mat> &input, vector<float> &output, const FCParams &params)
{
	assert(params.inputNums == input.size()*input[0].rows*input[0].cols);
	vector<float> inArray(params.inputNums);
	for (int map = 0; map < input.size(); map++)
	{
		for (int row = 0; row < input[map].rows; row++)
		{
			for (int col = 0; col < input[map].cols; col++)
			{
				inArray[map*input[map].rows*input[map].cols + row * input[map].cols + col]
					= input[map].at<float>(row, col);
			}
		}
	}
	fcLayer(inArray, output, params);
}

void softmaxLayer(vector<float> &input, vector<float> &output)
{
	float sum = 0;
	for (int n = 0; n < input.size(); n++)
	{
		output.push_back(exp(input[n]));
		sum += output[n];
	}
	for (int n = 0; n < input.size(); n++)
	{
		output[n] /= sum;
	}
}

int LeNet5_eval(Mat &input, const ConvLayers &convModel, const FCLayers &fcModel)
{
	vector<Mat> inVec;
	vector<Mat> conv1Out, conv2Out, pool1Out, pool2Out;
	vector<float> fc1Out, fc2Out, prob;

	inVec.push_back(input);

	convLayer(inVec, conv1Out, convModel.layerParams[0]);
	maxPoolLayer(conv1Out, pool1Out, 2, 2);
	convLayer(pool1Out, conv2Out, convModel.layerParams[1]);
	maxPoolLayer(conv2Out, pool2Out, 2, 2);
	fcLayer(pool2Out, fc1Out, fcModel.layerParams[0]);
	reluLayer(fc1Out);
	fcLayer(fc1Out,fc2Out, fcModel.layerParams[1]);
	softmaxLayer(fc2Out, prob);

	cout << "----------Probabilities Output---------" << endl;
	for (int p = 0; p < prob.size; p++)
	{
		cout << " p: " << prob[p] << endl;
	}
	cout << endl << endl;
	return distance(prob.begin(), max_element(prob.begin(), prob.end()));

}

