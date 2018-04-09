/*******************************************************************************************
 * Filename: ui_train.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the user interface for training a Neuronmancer model.
 *******************************************************************************************
 */

/* ui_train method - user interface for training a model */
void ui_train() {
    // declare helper variables for ui_create
    char inputBuffer[MAXINPUT]; // stores the user's input (gets recycled a lot)
    int tempInt -1; // stores int input from user (used for determining whether to run on host or GPU device)
    int myPatience = 2; // stores the amount of patience I have for the user's nonsense

    // declare variables needed to store the model's structure / training information
    float learningRate; // the rate that our network will learn
    unsigned int epochs; // the number of epochs for training (in a single epoch: see all training samples then use deltas for weight/bias update)
    unsigned int numberOfLayers; // the total number of layers in the network
    unsigned int numberOfNeuronsTotal; // the total number of neurons in the network
    unsigned int numberOfWeightsTotal; // the total number of weights in the network
    unsigned int* numberOfNeuronsInLayer; // the total number of neurons in each layer (1d array of size numberOfLayers)
    unsigned int* numberOfWeightsInFrontOfLayer; // the number of weights between each layer (1d array of size numberOfLayers)
    unsigned int* indexOfFirstNeuronInLayer; // the indexes of each layer's first neuron value (1d array of size numberOfLayers)
    unsigned int* indexOfFirstWeightInFrontOfLayer; // the indexes of the first weight value in front of each layer (1d array of size numberOfLayers)
    unsigned int numberOfTrainingSamples = 0; // the number of training samples in the MNIST training set

    // declare variables needed to store important model values and MNIST training set values
    float* neuronDeltas; // the delta value for each neuron (used to update weights / biases) 
    float* neurons; // the neuron values of the neural network (1d array of size numberOfNeuronsTotal)
    float* weights; // the weight values of the neural network (1d array of size numberOfWeightsTotal)
    float* biases; // the biases values of the neural network (1d array of size numberOfNeuronsTotal)
    float* expected; // the expected output values of a single sample (1D array of size numberOfNeuronsInLayer[numberOfLayers-1])
    char* trainingLabels; // the labels of each training sample (1D array of size numberOfTrainingSamples)
    unsigned char* trainingData; // the pixel-values of all training samples (1d array of size numberOfTrainingSmaples * MNISTSAMPLEDATASIZE)


    // declare variables needed to store device copies of the important model values and MNIST training set values
    float* devNeuronDeltas; // device copy of neuronDeltas
    float* devNeurons; // device copy of neurons
    float* devWeights; // device copy of weights
    float* devBiases; // device copy of biases
    float* devExpected; // device copy of expected
    char* devTrainingLabels; // device copy of trainingLabels
    unsigned char* devTrainingData; // device copy of trainingData
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////BELOW IS UNDER CONSTRUCTION///////////////////////////////////////////////////////////
///                                                                                                                                 ///




///                                                                                                                                 ///
////////////////////////////////////////////////ABOVE IS UNDER CONSTRUCTION////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // free the chunks of device memory that were dynamically allocated by cudaMalloc
    cudaFree(devNeuronDeltas);
    cudaFree(devNeurons);
    cudaFree(devWeights);
    cudaFree(devBiases);
    cudaFree(devExpected);
    cudaFree(devTrainingLabels);
    cudaFree(devTrainingData);

    // free the chunks of host memory that were dynamically allocated by malloc
    free(numberOfNeuronsInLayer);
    free(numberOfWeightsInFrontOfLayer);
    free(indexOfFirstNeuronInLayer);
    free(indexOfFirstWeightInFrontOfLayer);
    free(neuronDeltas);
    free(neurons);
    free(weights);
    free(biases);
    free(expected);
    free(trainingLabels);
    free(trainingData);


}//end ui_train function

