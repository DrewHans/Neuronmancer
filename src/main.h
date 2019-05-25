/*******************************************************************************
 * Filename: main.h
 * Author: Drew Hans (github.com/drewhans555)
 * Description: main.c's header file - contains important #include, #define,
 *              typedef, and struct definitions. Also contains the prototypes
 *              for main.c.
 *******************************************************************************
 */

#ifndef MAIN_H
#define MAIN_H

// include standard C headers
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// include standard CUDA headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAXINPUT 32 // size of our buffer for user input

// define macros for working with the MNIST data set
#define MNIST_TRAINING_SET_IMAGES_PATH "mnist/train-images-idx3-ubyte"
#define MNIST_TRAINING_SET_LABELS_PATH "mnist/train-labels-idx1-ubyte"
#define MNIST_TRAINING_SET_SIZE 60000 // number of MNIST training set samples

#define MNIST_TESTING_SET_IMAGES_PATH "mnist/t10k-images-idx3-ubyte"
#define MNIST_TESTING_SET_LABELS_PATH "mnist/t10k-labels-idx1-ubyte"
#define MNIST_TESTING_SET_SIZE 10000 // number of MNIST testing set samples

#define MNIST_CLASSIFICATIONS 10 // possible classifications are digits 0-9

#define MNIST_IMG_WIDTH 28  // the width of an MNIST image in pixels
#define MNIST_IMG_HEIGHT 28 // the height of an MNIST image in pixels
#define MNIST_IMG_PIXELS (MNIST_IMG_WIDTH * MNIST_IMG_HEIGHT)

// define macros for the neural network
#define INPUT_LAYER_SIZE MNIST_IMG_PIXELS
#define HIDDEN_LAYER_SIZE 256 // can be any number of neurons, in theory
#define OUTPUT_LAYER_SIZE MNIST_CLASSIFICATIONS

#define LEARNING_RATE 0.03 // weight & bias backpropagation adjustment rate

#define EPOCHS 1 // the number of training cycles
// Note: 1 epochs => seeing all training samples once
// and updating the weights / biases after each sample

// define types and structs for MNIST objects
typedef uint8_t MNIST_Label;

typedef struct
{
    uint8_t pixel[MNIST_IMG_PIXELS];
} MNIST_Image;

// ImageFileHeader format is defined here: http://yann.lecun.com/exdb/mnist/
typedef struct
{
    uint32_t magicNumber;
    uint32_t maxImages;
    uint32_t imgWidth;
    uint32_t imgHeight;
} MNIST_ImageFileHeader;

// LabelFileHeader format is defined here: http://yann.lecun.com/exdb/mnist/
typedef struct
{
    uint32_t magicNumber;
    uint32_t maxImages;
} MNIST_LabelFileHeader;

// define types and structs for neural network objects

// a single hidden-layer neuron
typedef struct
{
    // holds all weights connecting the individual inputs to the hidden neuron
    float weight[INPUT_LAYER_SIZE];

    // holds the weighted summation of all
    // InputLayer->input[i] * weight[i]
    // for each i in range INPUT_LAYER_SIZE
    float weightedSum;

    // holds the bias of the hidden neuron
    float bias;

    // holds the output of the hidden neuron after activation
    float output;

    // holds the delta value of the hidden neuron during backpropagation
    float delta;
} HLNeuron;

// a single output-layer neuron
typedef struct
{
    // holds all weights connecting the individual inputs to the output neuron
    float weight[HIDDEN_LAYER_SIZE];

    // holds the weighted summation of all
    // HiddenLayer->hNeuron[i].output * weight[i]
    // for each i in range HIDDEN_LAYER_SIZE
    float weightedSum;

    // holds the bias of the output neuron
    float bias;

    // holds the output of the output neuron after activation
    float output;

    // holds the delta value of the output neuron during backpropagation
    float delta;
} OLNeuron;

typedef struct
{
    // holds all the pixel values of a single MNIST image
    uint8_t input[INPUT_LAYER_SIZE];
} InputLayer;

typedef struct
{
    // holds all the hidden-layer neurons
    HLNeuron hNeuron[HIDDEN_LAYER_SIZE];
} HiddenLayer;

typedef struct
{
    // holds all the output-layer neurons
    OLNeuron oNeuron[OUTPUT_LAYER_SIZE];
} OutputLayer;

typedef struct
{
    int value[OUTPUT_LAYER_SIZE];
} ExpectedOutput;

// define types and structs for time collecting objects
typedef struct
{
    double averageFeedforwardTime;
    double averageBackpropagationTime;
    double averageUpdateTime;
    double averageEpochTime;
} TimeCollector;

// include Neuronmancer headers
#include "functions_mnist.h"
#include "functions_core.h"
#include "functions_cuda.h"

// include Neuronmancer source code
#include "functions_mnist.c"
#include "functions_core.c"
#include "functions_cuda.cu" // comment this line out if you don't want CUDA

// define function prototypes for main.c
void evaluate(InputLayer *il, HiddenLayer *hl, OutputLayer *ol);
void printConfusionMatrix(const int *cm);
void onInvalidInput(const int myPatience);

#endif /* MAIN_H */
