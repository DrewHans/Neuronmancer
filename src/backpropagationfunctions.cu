/*******************************************************************************************
 * Filename: backpropagationfunctions.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the host and device backpropagation functions.
 *******************************************************************************************
 */

/*
 * backpropagateErrorsKernel
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params:
 */
__global__ void backpropagateErrorsKernel() {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < numberOfNeuronsInLayer) {
        for (int w = 0; w < weightsInLayer; w += numberOfNeuronsInCurrentLayer) {
            int errortemp += (devWeights[firstWeightInPrevLayer + w] * devNeuronErrors[firstNeuronInPrevLayer + w]);
            devNeuronErrors[firstNeuronInCurrentLayer + id] = errortemp * sigmoidDerivative(devNeurons[firstNeuronInCurrentLayer + id]);
        }
    }
}//end backpropagate errors kernel

/*
 * weightUpdateKernel
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params:
 */
__global__ void weightUpdateKernel() {
    int neuronId = blockIdx.x;
    int weightId = threadIdx.x;
    if (neuronId < numberOfNeuronsInLeftLayer && weightId < numberOfWeightsBetweenLayers) {
        devWeights[firstWeightInBetweenLayers + numNeuronsInRightLayer*neuronId + weightId] += (learningRate * devNeuronErrors[firstNeuronInLeftLayer + neuronId] * devNeurons[firstNeuronInLeftLayer + neuronId]);
    }
}//end weight update kernel

/*
 * backpropagateWithDevice method
 */
void backpropagateWithDevice() {
#ifdef DEBUG
    printf("Entering backpropagateWithDevice method.\n");
#endif

    int numBlocks = 5;
    int threadsPerBlock = 32;

    // for each node in the output layer, calculate the output error
    //costFunctionKernel<<<numBlocks, threadsPerBlock>>>(devExpectedOutput, devNeurons, devNeuronErrors, neuronIndexStart, numberOfNeuronsInLayer);

    // for each layer l between output and input, visit in reverse order, backpropagate error values and update weights
    int outputLayerIndex = numberOfLayers-1;
    for (int l = outputLayerIndex - 1; l > 0; l--) {
        //backpropagateErrorsKernel<<<numBlocks, threadsPerBlock>>>();
        //weightUpdateKernel<<<numNeuronsInLayer, numNeuronsInPrevLayer>>>();
    }

#ifdef DEBUG
    printf("Leaving backpropagateWithDevice method.\n");
    printf("\n");
#endif
}//end backpropagateWithDevice method


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
void backpropagateWithHost(double* expectedOutput, double* neurons, double* weights, double* neuronErrors, int numberOfLayers, int* neuronsPerLayer, int* weightsPerLayer, int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer, double learningRate) {
#ifdef DEBUG
    printf("Entering backpropagate method.\n");
#endif

    // for each node in the output layer, calculate the output error
    int outputLayerIndex = numberOfLayers-1;
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
            for (int w = 0; w < weightsPerLayer[l+1]; w = w + neuronsPerLayer[l]) {
                errortemp = errortemp + (weights[firstWeightIndexPerLayer[l+1] + w] * neuronErrors[firstNeuronIndexPerLayer[l+1] + w]);
            }
            neuronErrors[firstNeuronIndexPerLayer[l] + n] = errortemp * sigmoidDerivative(neurons[firstNeuronIndexPerLayer[l] + n]);
        }
    }

    // for each layer l after input layer, update the weights in the layer
    for (int l = 1; l < numberOfLayers; l++) {
        // for each neuron in layer l
        for (int n = 0; n < neuronsPerLayer[l]; n++) {
            for (int w = 0; w < neuronsPerLayer[l-1]; w++) {
                weights[firstWeightIndexPerLayer[l] + neuronsPerLayer[l-1]*n + w] += (learningRate * neuronErrors[firstNeuronIndexPerLayer[l] + n] * neurons[firstNeuronIndexPerLayer[l] + n]);
            }
        }
    }

#ifdef DEBUG
    printf("Leaving backpropagate method.\n");
    printf("\n");
#endif
}//end backpropagateWithHost method

