//Definitions for LeNet-5 Model
#ifndef __LENET5_MODEL_H_
#define __LENET5_MODEL_H_

#include <stdio.h>

/* 
 *Params of CONV1 Layer
*/
#define CONV1_INPUTS_NUMS  		1
#define CONV1_OUTPUTS_NUMS 		20
#define CONV1_FILTER_HEIGHT		5
#define CONV1_FILTER_WIDTH		5
//Weights And Biases, exported from caffe
extern const float conv1_weights[CONV1_OUTPUTS_NUMS][CONV1_INPUTS_NUMS*CONV1_FILTER_WIDTH*CONV1_FILTER_HEIGHT];
extern const float conv1_bias[CONV1_OUTPUTS_NUMS];


/*
 * Params of CONV2 Layer
 */
#define CONV2_INPUTS_NUMS 		20
#define CONV2_OUTPUTS_NUMS		50
#define CONV2_FILTER_HEIGHT		5
#define CONV2_FILTER_WIDTH		5
//Weights And Biases, exported from caffe
extern const float conv2_weights[CONV2_OUTPUTS_NUMS][CONV2_INPUTS_NUMS*CONV2_FILTER_HEIGHT*CONV2_FILTER_WIDTH];
extern const float conv2_bias[CONV2_OUTPUTS_NUMS];

/*
 * Params of FC Layer
 */
#define FC1_INPUTS_NUMS			800
#define FC1_OUTPUTS_NUMS		500
extern const float fc1_weights[FC1_OUTPUTS_NUMS][FC1_INPUTS_NUMS];
extern const float fc1_bias[FC1_OUTPUTS_NUMS];

/* 
 * Params of FC2 Layer
 */
 #define FC2_INPUTS_NUMS		599
 #define FC2_OUTPUTS_NUMS		10
 extern const float fc2_weights[FC2_OUTPUTS_NUMS][FC2_INPUTS_NUMS];
 extern const float fc2_bias[FC2_OUTPUTS_NUMS];	


#endif