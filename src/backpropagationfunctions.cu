/*******************************************************************************************
 * Filename: backpropagationfunctions.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the host and device backpropagation functions.
 *******************************************************************************************
 */

/*
 * backpropagateErrorsKernel
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devNeurons - a pointer to an array of double values on device (the neuron values)
 * @params: devWeights - a pointer to an array of double values on device (the weight values)
 * @params: devNeuronErrors - a pointer to an array of double values on device (the deltas for each neuron)
 * @params: numberOfNeuronsInLeftLayer - the number of neurons in the layer left of the weights
 * @params: numberOfWeightsBetweenLayers - the number of weights between the layers
 * @params: indexOfFirstNeuronInLeft - the index of left layer's first neuron
 * @params: indexOfFirstNeuronInRight - the index of right layer's first neuron
 * @params: indexOfFirstWeight - the index of the first weight between layers
 */
__global__ void backpropagateErrorsKernel(double* devNeurons, double* devWeights, double* devNeuronErrors, int numberOfNeuronsInLeftLayer,
        int numberOfWeightsBetweenLayers, int indexOfFirstNeuronInLeft, int indexOfFirstNeuronInRight, int indexOfFirstWeight) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < numberOfNeuronsInLeftLayer) {
        int errortemp = 0;
        for (int w = 0; w < numberOfWeightsBetweenLayers; w += numberOfNeuronsInLeftLayer) {
            errortemp = errortemp + (devWeights[indexOfFirstWeight + w] * devNeuronErrors[indexOfFirstNeuronInRight + w]);
            devNeuronErrors[indexOfFirstNeuronInLeft + id] = errortemp * sigmoidDerivative(devNeurons[indexOfFirstNeuronInLeft + id]);
        }
    }
} //end backpropagate errors kernel

/*
 * weightUpdateKernel
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params:
 * @params: devNeurons - a pointer to an array of double values on device (the neuron values)
 * @params: devWeights - a pointer to an array of double values on device (the weight values)
 * @params: devNeuronErrors - a pointer to an array of double values on device (the deltas for each neuron)
 * @params: numberOfNeuronsInLeftLayer - the number of neurons in the layer left of the weights
 * @params: numberOfNeuronsInRightLayer - the number of neurons in the layer right of the weights
 * @params: numberOfWeightsBetweenLayers - the number of weights between the layers
 * @params: indexOfFirstNeuronInLeft - the index of left layer's first neuron
 * @params: indexOfFirstWeight - the index of the first weight between layers
 * @params: learningRate - the rate at which we want our network to make adjustments to the weights
 */
__global__ void weightUpdateKernel(double* devNeurons, double* devWeights, double* devNeuronErrors, int numberOfNeuronsInLeftLayer,
        int numberOfNeuronsInRightLayer, int numberOfWeightsBetweenLayers, int indexOfFirstNeuronInLeft, int indexOfFirstWeight, double learningRate) {
    if (blockIdx.x < numberOfNeuronsInLeftLayer && threadIdx.x < numberOfWeightsBetweenLayers) {
        int weightIndex = indexOfFirstWeight + numberOfNeuronsInRightLayer * blockIdx.x + threadIdx.x;
        int neuronIndex = indexOfFirstNeuronInLeft + blockIdx.x;
        devWeights[weightIndex] = devWeights[weightIndex] + (learningRate * devNeuronErrors[neuronIndex] * devNeurons[neuronIndex]);
    }
} //end weight update kernel

/*
 * backpropagateWithDevice method
 * @params: devExpectedOutput - the expected output values on device (needed to calculate output layer delta)
 * @params: devNeurons - a pointer to an array of double values on device (the neuron values)
 * @params: devWeights - a pointer to an array of double values on device (the weight values)
 * @params: devNeuronErrors - a pointer to an array of double values on device (the deltas for each neuron)
 * @params: numberOfLayers - the total number of layers in our artificial neural network
 * @params: neuronsPerLayer - a pointer to an array of int values (the number of neurons in each layer)
 * @params: weightsPerLayer - a pointer to an array of int values (the number of weights in each layer)
 * @params: firstNeuronIndexPerLayer - a pointer to an array of int values (the indexes of each layer's first neuron)
 * @params: firstWeightIndexPerLayer - a pointer to an array of int values (the indexes of each layer's first weight)
 * @params: learningRate - the rate at which we want our network to make adjustments to the weights
 */
void backpropagateWithDevice(double* devExpectedOutput, double* devNeurons, double* devWeights, double* devNeuronErrors, int numberOfLayers,
        int* neuronsPerLayer, int* weightsPerLayer, int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer, double learningRate) {
#ifdef DEBUG
    printf("Entering backpropagateWithDevice method.\n");
#endif

    // use getDeviceProperties helper function to determine the numBlocks and threadsPerBlock before launching CUDA Kernels
    int numBlocks = 5; // set 5 as default, should be equal to the number of SMs on the GPU device
    int threadsPerBlock = 32; // set 32 as default, should be equal to the warpsize on the GPU device
    getDeviceProperties(&numBlocks, &threadsPerBlock);

    // for each node in the output layer, calculate the output error (spawn 1 thread for each neuron in the output layer)
    int outputLayerIndex = numberOfLayers - 1;
    costFunctionKernel<<<numBlocks, threadsPerBlock>>>(devExpectedOutput, devNeurons, devNeuronErrors, firstNeuronIndexPerLayer[outputLayerIndex],
            neuronsPerLayer[outputLayerIndex]);

    // for each layer l between output and input, visit in reverse order, backpropagate error values and update weights
    for (int l = outputLayerIndex - 1; l > 0; l--) {
        // for each node in layer l, backpropagate the error from layer l+1 (spawn 1 thread for each neuron in layer l)
        backpropagateErrorsKernel<<<numBlocks, threadsPerBlock>>>(devNeurons, devWeights, devNeuronErrors, neuronsPerLayer[l], weightsPerLayer[l + 1],
                firstNeuronIndexPerLayer[l], firstNeuronIndexPerLayer[l + 1], firstWeightIndexPerLayer[l + 1]);

        // spawn 1 block for each neuron in layer l and, in each block, spawn 1 thread for each neuron in layer l+1
        weightUpdateKernel<<<neuronsPerLayer[l], neuronsPerLayer[l + 1]>>>(devNeurons, devWeights, devNeuronErrors, neuronsPerLayer[l], neuronsPerLayer[l + 1],
                weightsPerLayer[l + 1], firstNeuronIndexPerLayer[l], firstNeuronIndexPerLayer[l + 1], learningRate);
    }

#ifdef DEBUG
    printf("Leaving backpropagateWithDevice method.\n\n");
#endif
} //end backpropagateWithDevice method

/*
 * backpropagateWithHost method
 * @params: expectedOutput - the expected output values (needed to calculate output layer delta)
 * @params: neurons - a pointer to an array of double values (the neuron values)
 * @params: weights - a pointer to an array of double values (the weight values)
 * @params: neuronErrors - a pointer to an array of double values (the deltas for each neuron)
 * @params: numberOfLayers - the total number of layers in our artificial neural network
 * @params: neuronsPerLayer - a pointer to an array of int values (the number of neurons in each layer)
 * @params: weightsPerLayer - a pointer to an array of int values (the number of weights in each layer)
 * @params: firstNeuronIndexPerLayer - a pointer to an array of int values (the indexes of each layer's first neuron)
 * @params: firstWeightIndexPerLayer - a pointer to an array of int values (the indexes of each layer's first weight)
 * @params: learningRate - the rate at which we want our network to make adjustments to the weights
 */
void backpropagateWithHost(double* expectedOutput, double* neurons, double* weights, double* neuronErrors, int numberOfLayers, int* neuronsPerLayer,
        int* weightsPerLayer, int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer, double learningRate) {
#ifdef DEBUG
    printf("Entering backpropagate method.\n");
#endif

    // for each node in the output layer, calculate the output error
    int outputLayerIndex = numberOfLayers - 1;
    double errortemp = 0.0;
    for (int i = 0; i < neuronsPerLayer[outputLayerIndex]; i++) {
        errortemp = costFunction(&expectedOutput[i], &neurons[firstNeuronIndexPerLayer[outputLayerIndex] + i]);
        neuronErrors[firstNeuronIndexPerLayer[outputLayerIndex] + i] = errortemp * sigmoidDerivative(neurons[firstNeuronIndexPerLayer[outputLayerIndex] + i]);
    }

    // clear errortemp
    errortemp = 0.0;

    // for each layer l between output and input, visit in reverse order and backpropagate error values
    for (int l = outputLayerIndex - 1; l > 0; l--) {
        // for each neuron in layer l
        for (int n = 0; n < neuronsPerLayer[l]; n++) {
            // for each connection between layer l and l+1
            for (int w = 0; w < weightsPerLayer[l + 1]; w = w + neuronsPerLayer[l]) {
                errortemp = errortemp + (weights[firstWeightIndexPerLayer[l + 1] + w] * neuronErrors[firstNeuronIndexPerLayer[l + 1] + w]);
            }
            neuronErrors[firstNeuronIndexPerLayer[l] + n] = errortemp * sigmoidDerivative(neurons[firstNeuronIndexPerLayer[l] + n]);
        }
    }

    // for each layer l after input layer, update the weights in the layer
    for (int l = 1; l < numberOfLayers; l++) {
        // for each neuron in layer l
        for (int n = 0; n < neuronsPerLayer[l]; n++) {
            for (int w = 0; w < neuronsPerLayer[l - 1]; w++) {
                int weightIndex = firstWeightIndexPerLayer[l] + neuronsPerLayer[l - 1] * n + w;
                int neuronIndex = firstNeuronIndexPerLayer[l] + n;
                weights[weightIndex] = weights[weightIndex] + (learningRate * neuronErrors[neuronIndex] * neurons[neuronIndex]);
            }
        }
    }

#ifdef DEBUG
    printf("Leaving backpropagate method.\n\n");
#endif
} //end backpropagateWithHost method
