#include <stdio.h>
#include <string>
#include <opencv2\opencv.hpp>
#include "lenet5.h"
using namespace std;
using namespace cv;

void print_help(char **argv) {
	printf("Usage : %s\n"
		"-m sample -i <image path>\n"
		"\tOR\t\n"
		"-m test -f <image list file> -d <image dir> [-n <no images to test>]\n", argv[0]);
}

int main(int argc, char **argv) {

	char * mode = NULL;
	char * imgName = NULL;
	char * imgListFile = NULL;
	char * imgDir = NULL;
	int noTestImgs = -1;
	if (argc == 1) {
		print_help(argv);
		return -1;
	}

	// parse arguments and decide the application mode.
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-m")) {
			mode = argv[++i];
		}
		else if (!strcmp(argv[i], "-i")) {
			imgName = argv[++i];
		}
		else if (!strcmp(argv[i], "-f")) {
			imgListFile = argv[++i];
		}
		else if (!strcmp(argv[i], "-d")) {
			imgDir = argv[++i];
		}
		else if (!strcmp(argv[i], "-n")) {
			noTestImgs = atoi(argv[++i]);
		}
	}

	// Model storage 
	ConvLayers convModel;
	FCLayers fcModel;

	// Model initialization
	intiLeNet5Model(convModel, fcModel);

	if (!strcmp(mode, "sample")) {

		Mat input = imread(imgName, IMREAD_GRAYSCALE);

		// Input normalization to make it in the range [0, 1], assuming the input image to be in the range [0, 255]
		Mat normInput;
		input.convertTo(normInput, CV_32F);
		normInput = normInput / 255;


		// Forward pass of the network
		cout << "Starting the forward pass...." << endl;
		int predNo = LeNet5_eval(normInput, convModel, fcModel);
		cout << "The digit in the image is = " << predNo << endl;

	}
	else if (!strcmp(mode, "test")) {
		cout << "********MNIST Test Mode*********" << endl;

		std::ifstream listFile;
		std::vector<std::string> testImageList;
		std::vector<int> targetLabels, predLabels;
		std::string csvLine, imgFile, label;

		// read image list file and target labels and store in a vector
		listFile.open(imgListFile);
		while (std::getline(listFile, csvLine)) {
			std::istringstream ss(csvLine);
			std::getline(ss, imgFile, ',');
			std::getline(ss, label, ',');
			testImageList.push_back(imgFile);
			targetLabels.push_back(atoi(label.c_str()));
		}
		cout << "No of test images = " << targetLabels.size() << endl;
		int pred;
		int misCount = 0;
		Mat input, normInput;

		// This is the directory containing all MNIST test images.
		std::string testImgDir(imgDir);
		if (noTestImgs < 0) {
			noTestImgs = targetLabels.size();
		}

		for (int im = 0; im < noTestImgs; im++) {
			imgFile = testImgDir + "/" + testImageList[im];
			cout << imgFile << endl;
			input = imread(imgFile, IMREAD_GRAYSCALE);

			// Input normalization to make it in the range [0, 1], assuming the input image to be in the range [0, 255]
			//Mat normInput;
			input.convertTo(normInput, CV_32F);
			normInput = normInput / 255;

			// Prediction of digit
			pred = LeNet5_eval(normInput, convModel, fcModel);

			// check if the computer got it right...
			if (pred != targetLabels[im]) {
				misCount++;
			}
		}
		cout << "No images misclassified = " << misCount << endl;
		cout << "Classification Error = " << float(misCount) / noTestImgs << endl;


	}
	else {
		cout << "Invalid application mode" << endl;
		return -1;
	}


	return 0;
}
