/*******************************************************************************************
 * Filename: main.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This program creates a simple feed-forward artificial neural network
 *              and trains it on the CPU or GPU. The user will input (1) the number
 *              of layers (not including the input layer, which is required), (2) the
 *              number of neurons in each layer (including the input layer), and (3)
 *              whether to run on the CPU or GPU
 *******************************************************************************************
 */

#define MAXINPUT 32
#define DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "./neuronmancer.h"

#include "./helperfunctions.cu"
#include "./readmodel.cu"
#include "./savemodel.cu"

#include "./loadinput.cu"

#include "./activationfunctions.cu"
#include "./combinationfunctions.cu"
#include "./costfunctions.cu"

#include "./feedforwardfunctions.cu"
#include "./backpropagationfunctions.cu"

#include "./ui_create.cu"
#include "./ui_train.cu"
#include "./ui_evaluate.cu"

/* main - the program starts here */
int main(int argc, char * argv[]) {
    printf("Starting %s...\n", argv[0]);
    printf("Lets create a simple artificial neural network!\n");
    ui_create();

    // declare our host variables
    char inputBuffer[MAXINPUT]; // store the user's input (gets recycled a lot)
/*    int numberOfLayers; // store the total number of layers in the network
    int numberOfNeuronsTotal; // store the total number of neurons in our neural network
    int numberOfWeightsTotal; // store the total number of weights in our neural network
    int* numberOfNeuronsPerLayer; // store the total number of neurons in each layer in our neural network in a 1d array of size numberOfLayers
    int* numberOfWeightsPerLayer; // store the total number of weights between each layer in our neural network in a 1d array of size numberOfLayers-1
    int* firstNeuronIndexPerLayer; // store the indexes of each layer's first neuron value
    int* firstWeightIndexPerLayer; // store the indexes of each layer's first weight value
    double* neurons; // store the neuron values of our neural network in a 1d array of size neuronSize (1d arrays are easy to work with in CUDA)
    double* weights; // store the weight values of our neural network in a 1d array of size weightSize (1d arrays are easy to work with in CUDA)
    double* outputExpected; // store the outputExpected output values for backpropagation
    double* neuronErrors; // store the error "cost" of each neuron during backpropagation
    char runOn; // store the user's choice of host or device
    int epochs = 5; // store the number of epochs for training
    double learningRate = 0.5; // store the rate that our network will learn

    // declare our device variables
    int* deviceNumberOfNeuronsPerLayer;
    int* deviceNumberOfWeightsPerLayer;
    double* deviceNeurons;
    double* deviceWeights;
    double* deviceWeightCosts;

    // declare our cudaStatus variable
    // cudaError_t cudaStatus;

    printf("For the following please enter a positive number with no spaces, commas, or decimal points and in less than 31 characters.\n");

    // get the number of layers in the ANN
    printf("How many hidden layers do you want this network to have?\n~");
    fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
    sscanf(inputBuffer, "%d", &numberOfLayers); // format and dump the user's input
    numberOfLayers = numberOfLayers + 2; // account for the input layer

    // dynamically allocate memory for our variables
    numberOfNeuronsPerLayer = (int *) malloc(numberOfLayers * sizeof(int)); //malloc allocates a chunk of host memory
    numberOfWeightsPerLayer = (int *) malloc(numberOfLayers * sizeof(int)); //malloc allocates a chunk of host memory
    firstNeuronIndexPerLayer = (int *) malloc(numberOfLayers * sizeof(int)); //malloc allocates a chunk of host memory
    firstWeightIndexPerLayer = (int *) malloc(numberOfLayers * sizeof(int)); //malloc allocates a chunk of host memory

    // get the number of neurons in input layer in the ANN
    printf("How many neurons do you want the input layer to have?\n~");
    fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
    sscanf(inputBuffer, "%d", &numberOfNeuronsPerLayer[0]); // format and dump the user's input
    for (int i = 1; i < numberOfLayers - 1; i++) {
        printf("How many neurons do you want hidden layer %d to have?\n~", i);
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
        sscanf(inputBuffer, "%d", &numberOfNeuronsPerLayer[i]); // format and dump the user's input
    }
    printf("How many neurons do you want the output layer to have?\n~");
    fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
    sscanf(inputBuffer, "%d", &numberOfNeuronsPerLayer[numberOfLayers - 1]); // format and dump the user's input

    printf("Do you want to run on the host CPU or device GPU?\nEnter h for host or d for device:\n~");
    fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
    sscanf(inputBuffer, "%c", &runOn); // format and dump the user's input

    // Calculate the number of neuron/weight values we need space for and also the first Neuron/Weight index for each layer
    firstNeuronIndexPerLayer[0] = 0;  // input layer's first neuron starts at 0
    firstWeightIndexPerLayer[0] = -1; // input layer has no weights, put -1 just for fun
    numberOfWeightsPerLayer[0] = 0;   // input layer has no weights
    numberOfNeuronsTotal = numberOfNeuronsPerLayer[0]; // start by counting the neurons in input layer
    numberOfWeightsTotal = 0; // input layer has no weights
    for (int i = 1; i < numberOfLayers; i++) {
        firstNeuronIndexPerLayer[i] = numberOfNeuronsTotal;
        firstWeightIndexPerLayer[i] = numberOfWeightsTotal;
        numberOfWeightsPerLayer[i] = numberOfNeuronsPerLayer[i - 1] * numberOfNeuronsPerLayer[i];
        numberOfNeuronsTotal = numberOfNeuronsTotal + numberOfNeuronsPerLayer[i];
        numberOfWeightsTotal = numberOfWeightsTotal + (numberOfWeightsPerLayer[i]);
    }

    // dynamically allocate memory to store the neuron values, weight values, and outputExpected output values
    neurons = (double*) malloc(numberOfNeuronsTotal * sizeof(double)); //malloc allocates a chunk of host memory
    weights = (double*) malloc(numberOfWeightsTotal * sizeof(double)); //malloc allocates a chunk of host memory
    neuronErrors = (double*) malloc(numberOfNeuronsTotal * sizeof(double)); //malloc allocates a chunk of host memory
    outputExpected = (double*) malloc(numberOfNeuronsPerLayer[numberOfLayers - 1] * sizeof(double)); //malloc allocates a chunk of host memory

    // initialize every neuron and weight value to zero (clean up any garbage we may have picked up)
    printf("Starting init step now...\n");
    initArrayToZeros(neurons, numberOfNeuronsTotal);
    printf("\n");
    initArrayToRandomDoubles(weights, numberOfWeightsTotal);

    printf("initNeurons & initWeights successful!\n\n");

    if (runOn == 'd') {
        printf("Allocating GPU device memory and copying host values over...\n");

        // allocate device memory for device variables and copy host values to device copies
        cudaMalloc((void **) &deviceNumberOfNeuronsPerLayer, numberOfLayers * sizeof(int)); //cudaMalloc allocates a chunk of device memory
        cudaMalloc((void **) &deviceNumberOfWeightsPerLayer, numberOfLayers * sizeof(int)); //cudaMalloc allocates a chunk of device memory
        cudaMalloc((void **) &deviceNeurons, (numberOfNeuronsTotal * sizeof(double))); //cudaMalloc allocates a chunk of device memory
        cudaMalloc((void **) &deviceWeights, (numberOfWeightsTotal * sizeof(double))); //cudaMalloc allocates a chunk of device memory
        cudaMalloc((void **) &deviceWeightCosts, (numberOfWeightsTotal * sizeof(double))); //cudaMalloc allocates a chunk of device memory
        cudaMemcpy(deviceNumberOfNeuronsPerLayer, numberOfNeuronsPerLayer, (numberOfLayers * sizeof(int)), cudaMemcpyHostToDevice); //cudaMemcpy copies host values to device copies
        cudaMemcpy(deviceNumberOfWeightsPerLayer, numberOfWeightsPerLayer, (numberOfLayers * sizeof(int)), cudaMemcpyHostToDevice); //cudaMemcpy copies host values to device copies
        cudaMemcpy(deviceNeurons, neurons, (numberOfNeuronsTotal * sizeof(double)), cudaMemcpyHostToDevice); //cudaMemcpy copies host values to device copies
        cudaMemcpy(deviceWeights, weights, (numberOfWeightsTotal * sizeof(double)), cudaMemcpyHostToDevice); //cudaMemcpy copies host values to device copies

        printf("Allocation successful!\n\n");
    }

    for (int i = 0; i < epochs; i++) {
        // HOST - LOADINPUT, FEEDFORWARD, & BACKPROPAGATE
        printf("Starting load input step now...\n");
        loadInput(neurons, numberOfNeuronsPerLayer[0]); // load some random input for feedforward testing
        printarray("neurons", neurons, numberOfNeuronsTotal);

        printf("Starting feedforward step now...\n");
        feedforwardWithHost(neurons, weights, numberOfLayers, numberOfNeuronsPerLayer, numberOfWeightsPerLayer, firstNeuronIndexPerLayer,
                firstWeightIndexPerLayer); // feed the input forward

        printf("Network state post feedforward:\n");
        printarray("neurons", neurons, numberOfNeuronsTotal);
        printarray("weights", weights, numberOfWeightsTotal);

        printf("Generating random training labels for testing backpropagation now...\n");
        loadInput(outputExpected, numberOfNeuronsPerLayer[numberOfLayers - 1]); // load some random input for backpropagation testing
        printarray("outputExpected", outputExpected, numberOfNeuronsPerLayer[numberOfLayers - 1]);

        printf("Starting backpropagation step now...\n");
        backpropagateWithHost(outputExpected, neurons, weights, neuronErrors, numberOfLayers, numberOfNeuronsPerLayer, numberOfWeightsPerLayer,
                firstNeuronIndexPerLayer, firstWeightIndexPerLayer, learningRate); // calculate and back propagate errors

        printf("Network state post backpropagation:\n");
        printarray("neurons", neurons, numberOfNeuronsTotal);
        printarray("weights", weights, numberOfWeightsTotal);
    }

    printf("Press enter to free dynamically allocated memory.\n~");
    fgets(inputBuffer, MAXINPUT, stdin); // read the user's input

    printf("Freeing dynamically allocated memory...");

    // free the chunks of host memory that were dynamically allocated by malloc
    free(numberOfNeuronsPerLayer);
    free(numberOfWeightsPerLayer);
    free(firstNeuronIndexPerLayer);
    free(firstWeightIndexPerLayer);
    free(neurons);
    free(weights);
    free(neuronErrors);
    free(outputExpected);

    if (runOn == 'd') {
        // free the chunks of device memory that were dynamically allocated by cudaMalloc
        cudaFree(deviceNumberOfNeuronsPerLayer);
        cudaFree(deviceNumberOfWeightsPerLayer);
        cudaFree(deviceNeurons);
        cudaFree(deviceWeights);
        cudaFree(deviceWeightCosts);
    }

    printf("Memory freed!\n");
*/
    printf("%s will now end. ", argv[0]);
    printf("Press enter to end.\n~");
    fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
} //end main method
