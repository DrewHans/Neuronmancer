/********************************************************************************
 * Filename: main.h
 * Author: Drew Hans (github.com/drewhans555)
 * Description: main.c's header file - contains important #include, #define,
 *              typedef, and struct definitions. Also contains the prototypes
 *              for main.c.
 ********************************************************************************
 */

#ifndef MAIN_H
#define MAIN_H

/////////////////////////////////////////////////////////////////////////////////
// include the standard C headers ///////////////////////////////////////////////
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/////////////////////////////////////////////////////////////////////////////////
// include the standard CUDA headers ////////////////////////////////////////////
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/////////////////////////////////////////////////////////////////////////////////
// define macros for our network parameters /////////////////////////////////////

#define IL_SIZE 784  // input-layer size - should be equal to the number of input values (28x28 for an MNIST image)
#define HL_SIZE 256 // hidden-layer size - should be equal to the number of hidden-layer neurons (can be any size, in theory)
#define OL_SIZE 10  // output-layer size - should be equal to the number of possible classifications (digits 0 -9 for an MNIST image)
#define LEARNING_RATE 0.03 // the rate of weight / bias adjustment after backpropagation
#define EPOCHS 1 // the number of training cycles (1 cycle => seeing all training samples once and updating the weights / biases after each sample)

#define MAXINPUT 32 // the size of our buffer for user input

/////////////////////////////////////////////////////////////////////////////////
// define macros for working with the MNIST data set ////////////////////////////

#define MNIST_TRAINING_SET_IMAGES_LOCATION "mnist/train-images-idx3-ubyte" // path to the file of MNIST training set images
#define MNIST_TRAINING_SET_LABELS_LOCATION "mnist/train-labels-idx1-ubyte"  // path to the file of MNIST training set labels

#define MNIST_TESTING_SET_IMAGES_LOCATION "mnist/t10k-images-idx3-ubyte"   // path to the file of MNIST testing set labels
#define MNIST_TESTING_SET_LABELS_LOCATION "mnist/t10k-labels-idx1-ubyte"    // path to the file of MNIST testing set labels

#define MNIST_CLASSIFICATIONS 10      // number of possible classifications for an MNIST image
#define MNIST_TRAINING_SET_SIZE 60000 // number of MNIST training samples
#define MNIST_TESTING_SET_SIZE 10000  // number of MNIST testing samples
#define MNIST_IMG_PIXELS 784          // number of pixels in an MNIST image
#define MNIST_IMG_WIDTH 28            // the width of an MNIST image in pixels
#define MNIST_IMG_HEIGHT 28           // the height of an MNIST image in pixels

/////////////////////////////////////////////////////////////////////////////////
// define types and structs for MNIST objects ///////////////////////////////////

typedef uint8_t MNIST_Label;

typedef struct {
    uint8_t pixel[MNIST_IMG_WIDTH * MNIST_IMG_HEIGHT];
} MNIST_Image;

// ImageFileHeader format is defined here: http://yann.lecun.com/exdb/mnist/
typedef struct {
    uint32_t magicNumber;
    uint32_t maxImages;
    uint32_t imgWidth;
    uint32_t imgHeight;
} MNIST_ImageFileHeader;

// LabelFileHeader format is defined here: http://yann.lecun.com/exdb/mnist/
typedef struct {
    uint32_t magicNumber;
    uint32_t maxImages;
} MNIST_LabelFileHeader;

/////////////////////////////////////////////////////////////////////////////////
// define types and structs for neural network objects //////////////////////////

typedef struct {
    float weight[IL_SIZE]; // holds all weights connecting the individual inputs to the hidden neuron
    float weightedSum;     // holds the weighted summation of all InputLayer->input[i] * weight[i] for each i in range IL_SIZE
    float bias;            // holds the bias of the hidden neuron
    float output;          // holds the output of the hidden neuron after activation
    float delta;           // holds the delta value of the hidden neuron during backpropagation
} HLNeuron;

typedef struct {
    float weight[HL_SIZE]; // holds all weights connecting the individual inputs to the output neuron
    float weightedSum;     // holds the weighted summation of all HiddenLayer->hNeuron[i].output * weight[i] for each i in range HL_SIZE
    float bias;            // holds the bias of the output neuron
    float output;          // holds the output of the output neuron after activation
    float delta;           // holds the delta value of the output neuron during backpropagation
} OLNeuron;

typedef struct {
    uint8_t input[IL_SIZE];
} InputLayer;

typedef struct {
    HLNeuron hNeuron[HL_SIZE];
} HiddenLayer;

typedef struct {
    OLNeuron oNeuron[OL_SIZE];
} OutputLayer;

typedef struct {
    int value[OL_SIZE];
} ExpectedOutput;

/////////////////////////////////////////////////////////////////////////////////
// define types and structs for time collecting objects /////////////////////////

typedef struct {
    double averageFeedforwardTime;     // holds the average time spent feedforwarding
    double averageBackpropagationTime; // holds the average time spent backpropagating
    double averageUpdateTime;          // holds the average time spent updating weights / biases
    double averageEpochTime;           // holds the average time spent in a single epoch
} TimeCollector;

/////////////////////////////////////////////////////////////////////////////////
// include Neuronmancer headers /////////////////////////////////////////////////
#include "functions_mnist.h"
#include "functions_core.h"
#include "functions_cuda.h"

/////////////////////////////////////////////////////////////////////////////////
// include Neuronmancer source code /////////////////////////////////////////////

#include "functions_mnist.c"
#include "functions_core.c"
#include "functions_cuda.cu" // comment this line out if you don't want CUDA

/////////////////////////////////////////////////////////////////////////////////
// define function prototypes for main.c ////////////////////////////////////////

void evaluate(InputLayer* il, HiddenLayer* hl, OutputLayer* ol);
void printConfusionMatrix(const int* cm);
void onInvalidInput(const int myPatience);

/////////////////////////////////////////////////////////////////////////////////

#endif /* MAIN_H */
