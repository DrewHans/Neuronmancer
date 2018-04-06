/********************************************************************************
 * Filename: main.h
 * Author: Drew Hans (github.com/drewhans555)
 * Description: Header file for main.cu
 ********************************************************************************
 */

#ifndef MAIN_H
#define MAIN_H

// include the standard C headers
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

// define macros for input buffers, debugging mode, etc.
#define MAXINPUT 32
#define DEBUG

// define macros for model_read.cu and model_save.cu
#define VALUEDELIM ','
#define MODELDIRECTORY "../nmModel"
#define MODELVALUESLOCATION "../nmModel/modelvalues.csv"
#define WEIGHTSFILELOCATION "../nmModel/weights.csv"
#define BIASESFILELOCATION "../nmModel/biases.csv"
#define EPOCHSFILELOCATION "../nmModel/epochs.txt"
#define LEARNINGRATEFILELOCATION "../nmModel/learningrate.txt"

// define macros for working with the MNIST data set
#define MNISTSAMPLEDATASIZE 784
#define MNISTTESTSETSIZE 10000
#define MNISTTRAININGSETSIZE 60000
#define MNISTTESTFILELOCATION "../mnist/mnist_test.csv"
#define MNISTTRAINFILELOCATION "../mnist/mnist_train.csv"

// define struct for using the stat command
struct stat st = { 0 };

// define function prototypes for activations.cu
__host__ __device__ double sigmoidFunction(double d);
__host__ __device__ double sigmoidDerivative(double d);
__host__ __device__ double tanhFunction(double d);
__host__ __device__ double tanhDerivative(double d);
__host__ __device__ double reluFunction(double d);
__host__ __device__ double reluDerivative(double d);
__global__ void sigmoidKernel(double* devNeurons, int neuronIndexStart, int numberOfNeuronsInLayer);
__global__ void reluKernel(double* devNeurons, int neuronIndexStart, int numberOfNeuronsInLayer);
__global__ void tanhKernel(double* devNeurons, int neuronIndexStart, int numberOfNeuronsInLayer);

// define function prototypes for backpropagation.cu

// define function prototypes for feedforward.cu

// define function prototypes for functions_misc.cu

// define function prototypes for functions_mnist.cu

// define function prototypes for model_read.cu

// define function prototypes for model_save.cu

// define function prototypes for ui_create.cu
void ui_create();

// define function prototypes for ui_evaluate.cu
void ui_evaluate();

// define function prototypes for ui_train.cu
void ui_train();

// include the CUDA headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// include the Neuronmancer src code files
#include "./functions_misc.cu"
#include "./functions_mnist.cu"
#include "./model_read.cu"
#include "./model_save.cu"
#include "./activations.cu"
#include "./backpropagation.cu"
#include "./feedforward.cu"
#include "./ui_create.cu"
#include "./ui_evaluate.cu"
#include "./ui_train.cu"

#endif /* MAIN_H */

