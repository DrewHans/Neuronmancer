/*******************************************************************************************
 * Filename: feedforwardfunctions.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the host and device feedforward functions.
 *******************************************************************************************
 */

/*
 * feedforwardWithDevice - propagates the inputs forward to compute the outputs
 * @params: devNeurons - a pointer to an array of double values (the neuron values) in device memory
 * @params: devWeights - a pointer to an array of double values (the weight values) in device memory
 * @params: numberOfLayers - the total number of layers in our artificial neural network
 * @params: neuronsPerLayer - a pointer to an array of int values (the number of neurons in each layer)
 * @params: weightsPerLayer - a pointer to an array of int values (the number of weights in each layer)
 * @params: firstNeuronIndexPerLayer - a pointer to an array of int values (the indexes of each layer's first neuron)
 * @params: firstWeightIndexPerLayer - a pointer to an array of int values (the indexes of each layer's first weight)
 */
void feedforwardWithDevice(double* devNeurons, double* devWeights, int numberOfLayers, int* numberOfNeuronsPerLayer, int* numberOfWeightsPerLayer,
        int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer) {
#ifdef DEBUG
    printf("Entering feedforwardWithDevice method.\n");
#endif
    int numBlocks = 5;
    int threadsPerBlock = 32;

    // go layer to layer and for each neuron in the current layer: spawn a thread, perform combination, sync threads, spawn a thread
    for (int l = 1; l < numberOfLayers; l++) {
        combinationFunctionKernel<<<numBlocks, threadsPerBlock>>>(devNeurons, devWeights, firstNeuronIndexPerLayer[l], firstNeuronIndexPerLayer[l - 1],
                firstWeightIndexPerLayer[l], numberOfNeuronsPerLayer[l], numberOfNeuronsPerLayer[l - 1]);
        cudaDeviceSynchronize(); // tell host to wait for device to finish previous kernel
        sigmoidKernel<<<numBlocks, threadsPerBlock>>>(devNeurons, firstNeuronIndexPerLayer[l], numberOfNeuronsPerLayer[l]);
    }

#ifdef DEBUG
    printf("Leaving feedforwardWithDevice method.\n\n");
#endif
} //end feedforwardWithDevice method

/*
 * feedforwardWithHost - propagates the inputs forward to compute the outputs
 * @params: neurons - a pointer to an array of double values (the neuron values)
 * @params: weights - a pointer to an array of double values (the weight values)
 * @params: numberOfLayers - the total number of layers in our artificial neural network
 * @params: neuronsPerLayer - a pointer to an array of int values (the number of neurons in each layer)
 * @params: weightsPerLayer - a pointer to an array of int values (the number of weights in each layer)
 * @params: firstNeuronIndexPerLayer - a pointer to an array of int values (the indexes of each layer's first neuron)
 * @params: firstWeightIndexPerLayer - a pointer to an array of int values (the indexes of each layer's first weight)
 */
void feedforwardWithHost(double* neurons, double* weights, int numberOfLayers, int* neuronsPerLayer, int* weightsPerLayer, int* firstNeuronIndexPerLayer,
        int* firstWeightIndexPerLayer) {
#ifdef DEBUG
    printf("Entering feedforwardWithHost method.\n");
#endif

    // go layer to layer
    for (int i = 1; i < numberOfLayers; i++) {
        // go neuron to neuron in layer i
        for (int j = 0; j < neuronsPerLayer[i]; j++) {
            combinationFunction(neurons, weights, firstNeuronIndexPerLayer[i] + j, firstNeuronIndexPerLayer[i - 1], firstWeightIndexPerLayer[i] + j,
                    neuronsPerLayer[i - 1]);
            sigmoidFunction(neurons[firstNeuronIndexPerLayer[i] + j]);
            printf("neurons[%d]=%f\n", (firstNeuronIndexPerLayer[i] + j), neurons[firstNeuronIndexPerLayer[i] + j]);
        }
    }

#ifdef DEBUG
    printf("Leaving feedforwardWithHost method.\n\n");
#endif
} //end feedforwardWithHost method
