#include "lenet5_model.h"
//Define Layers Structures
typedef struct ConvWeightBias
{
	vector<vector<Mat> > W;
	vector<float> bias;
	int inputNums;
	int outputNums;
	int filterW;
	int filterH;
} ConvParams;

typedef struct ConvLayers
{
	vector<ConvParams> layerParams;
	int layersNum;
} ConvLayers;

typedef struct FCWeightBias
{
	Mat W;
	vector<float> bias;
	int inputNums;
	int outputNums;
} FCParams;

typedef struct FCLayers
{
	vector<FCParams> layerParams;
	int layersNum;
} FCLayers;

void intiLeNet5Model(ConvLayers &conv, FCLayers &fc);
int LeNet5_eval(Mat &input, const ConvLayers &convModel, const FCLayers &fcModel);