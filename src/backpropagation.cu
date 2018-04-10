/*******************************************************************************************
 * Filename: feedforward.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the host and device backpropagation functions, update
 *              functions for weights and biases, cudaKernels for calculating the deltas 
 *              and error amount for each neuron in each layer in the network, cudaKernels 
 *              updating the weights and biases.
 *******************************************************************************************
 */

/*
 * backpropagationUsingHost - calculate then backpropagate the output-layer's delta through the network using CPU
 * @params: neuronDeltas - a float pointer-pointer to the chunk of memory containing the delta values for each neuron
 * @params: expected - a float pointer to the chunk of memory containing the expected output neuron values (from training labels)
 * @params: neurons - a float pointer to the chunk of memory containing the neuron values
 * @params: weights - a float pointer to the chunk of memory containing the weight values
 * @params: biases - a float pointer to the chunk of memory containing the biases values
 * @params: numberOfLayers - the int number of layers in the network (index 0 => input, index numberOfLayers-1 => output)
 * @params: numberOfNeuronsInLayer - an int pointer to a chunk of memory containing the number of neurons in each layer
 * @params: numberOfWeightsInFrontOfLayer - an int pointer to a chunk of memory containing the number of weights in front of each layer
 * @params: indexOfFirstNeuronInLayer - an int pointer to a chunk of memory containing the indexes of the first neurons in each layer
 * @params: indexOfFirstWeightInFrontOfLayer - an int pointer to a chunk of memory containing the indexes of the first weight in front of each layer
 */
void backpropagationUsingHost(float** neuronDeltas, const float* expected, const float* neurons, const float* weights, const float* biases, 
                              const unsigned int numberOfLayers, const unsigned int* numberOfNeuronsInLayer, const unsigned int* numberOfWeightsInFrontOfLayer,
                              const unsigned int* indexOfFirstNeuronInLayer, const unsigned int* indexOfFirstWeightInFrontOfLayer) {
    unsigned int indexOfFirstOutputNeuron = indexOfFirstNeuronInLayer[numberOfLayers-1]; // start at this neuron for output-layer

    // calculate the deltas for each neuron no in the output-layer
    for (int no = 0; no < numberOfNeuronsInLayer[numberOfLayers-1]; no++) {
        // note: we add whatever is in neuronDeltas[no's index] to the delta value (this is so we have a summation of all deltas 
        //       for later... as long as we don't forget to divide by the number of samples we trained on it should be fine)
        (*neuronDeltas)[indecxOfFirstOutputNeuron + no] += (quadraticCostDerivative(expected[no], neurons[indexOfFirstOutputNeuron + no]) * sigmoidDerivative(neurons[indexOfFirstOutputNeuron + no]));
    }

    // for each layer l between output and input (non-inclusive) visit in reverse order and backpropagate error values from right to left
    for (int l = numberOfLayers - 2; l > 0; l--) {
        unsigned int numberOfNeuronsInLeft = numberOfNeuronsInLayer[l]; // left-layer size
        unsigned int numberOfNeuronsInRight = numberOfNeuronsInLayer[l+1]; // right-layer size
        unsigned int numberOfWeights = numberOfWeightsInFrontOfLayer[l+1]; // the number of weights between left and right

        unsigned int indexOfFirstLeftNeuron = indexOfFirstNeuronInLayer[l]; // start at this neuron for left-layer
        unsigned int indexOfFirstRightNeuron = indexOfFirstNeuronInLayer[l+1]; // start at this neuron for right-layer
        unsigned int indexOfFirstWeight = indexOfFirstWeightInFrontOfLayer[l+1]; // start at this weight 

        // for each neuron nl in left-layer:
        // calculate the errorSum of all right-layer connections, then use the errorSum to calculate nl's delta value and add it to neuronDeltas[nl's index]
        for (int nl = 0; nl < numberOfNeuronsInLayer[l]; nl++) {
            float errorSum = 0.0; // store neuron nl's error sum of all right-layer connections

            // for each neuron nr in right-layer
            for (int nr = 0; nr < numberOfNeuronsInRight; nr++) {
                // add right-layer neuron nr's delta and it's weighted link to left-layer neuron nl to nl's error sum
                errorSum += (((*neuronDeltas)[indexOfFirstRightNeuron + nr]) * weights[indexOfFirstWeight + nr*numberOfNeuronsInLeft]);
            }

            // calculate nl's delta and store in neuronDeltas[nl's index]
            // note: we add whatever is in neuronDeltas[nl's index] to the delta value (this is so we have a summation of all deltas in a given epoch)
            (*neuronDeltas)[indexOfFirstLeftNeuron + nl] += errorSum * sigmoidDerivative(neurons[indexOfFirstLeftNeuron + nl]);
        }
    }
}//end backpropagationUsingHost function

/*
 * backpropagationUsingDevice - calculate then backpropagate the output-layer's delta through the network using GPU
 * @params: devNeuronDeltas - device copy of float* neuronDeltas
 * @params: devExpected - device copy of float* expected
 * @params: devNeurons - device copy of float* neurons
 * @params: devWeights - device copy of float* weights
 * @params: devBiases - device copy of float* biases
 * @params: numberOfLayers - the int number of layers in the network (index 0 => input, index numberOfLayers-1 => output)
 * @params: numberOfNeuronsInLayer - an int pointer to a chunk of memory containing the number of neurons in each layer
 * @params: numberOfWeightsInFrontOfLayer - an int pointer to a chunk of memory containing the number of weights in front of each layer
 * @params: indexOfFirstNeuronInLayer - an int pointer to a chunk of memory containing the indexes of the first neurons in each layer
 * @params: indexOfFirstWeightInFrontOfLayer - an int pointer to a chunk of memory containing the indexes of the first weight in front of each layer
 */
void backpropagationUsingDevice(float* devNeuronDeltas, float* devExpected, float* devNeurons, float* devWeights, float* devBiases, 
                                const unsigned int numberOfLayers, const unsigned int* numberOfNeuronsInLayer, const unsigned int* numberOfWeightsInFrontOfLayer,
                                const unsigned int* indexOfFirstNeuronInLayer, const unsigned int* indexOfFirstWeightInFrontOfLayer) {
    unsigned int numberOfNeuronsInOutput = numberOfNeuronsInLayer[numberOfLayers-1];
    unsigned int indexOfFirstOutputNeuron = indexOfFirstNeuronInLayer[numberOfLayers-1]; // start at this neuron for output-layer

    // use getDeviceProperties helper function to get GPU device information
    unsigned int numberOfSMs = 0; // the number of SMs on the device (1 SM can process 1 block at a time)
    unsigned int warpsize = 0; // the number of threads that an SM can manage at one time
    getDeviceProperties(&numberOfSMs, &warpsize); 

    // set blocks and threads to a size that will fully utilize the GPU (overkill, I know, but we're going for performance here)
    unsigned int blocks = numberOfSMs; // should be equal to the number of SMs on the GPU device after getDeviceProperties
    unsigned int threads = warpsize; // should be equal to the warpsize on the GPU device after getDeviceProperties
    
    // double or devide the number of threads until we have a number close to the number of neurons in output-layer
    threads = getOptimalThreadSize(blocks, threads, numberOfNeuronsInOutput, warpsize);

    // calculate the deltas for each neuron no in the output-layer
    cudaKernel_CalculateOutputLayerDeltas<<<blocks, threads>>>(devNeuronDeltas, devExpected, devNeurons, numberOfNeuronsInOutput, indexOfFirstOutputNeuron);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        onCudaKernelLaunchFailure("cudaKernel_CalculateOutputLayerDeltas", cudaStatus);
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        onCudaDeviceSynchronizeError("cudaKernel_CalculateOutputLayerDeltas", cudaStatus);
    }

    // for each layer l between output and input (non-inclusive) visit in reverse order and backpropagate error values from right to left
    for (int l = numberOfLayers - 2; l > 0; l--) {
        unsigned int numberOfNeuronsInLeft = numberOfNeuronsInLayer[l]; // left-layer size
        unsigned int numberOfNeuronsInRight = numberOfNeuronsInLayer[l+1]; // right-layer size
        unsigned int numberOfWeights = numberOfWeightsInFrontOfLayer[l+1]; // the number of weights between left and right

        unsigned int indexOfFirstLeftNeuron = indexOfFirstNeuronInLayer[l]; // start at this neuron for left-layer
        unsigned int indexOfFirstRightNeuron = indexOfFirstNeuronInLayer[l+1]; // start at this neuron for right-layer
        unsigned int indexOfFirstWeight = indexOfFirstWeightInFrontOfLayer[l+1]; // start at this weight 

        // double or devide the number of threads until we have a number close to the number of neurons in output-layer
        threads = getOptimalThreadSize(blocks, threads, numberOfNeuronsInLeft, warpsize);

        // for each neuron nl in left-layer:
        // calculate the errorSum of all right-layer connections, then use the errorSum to calculate nl's delta value and add it to neuronDeltas[nl's index]
        cudaKernel_CalculateLeftLayerDeltas(devNeuronDeltas, devExpected, devNeurons, devWeights, numberOfNeuronsInLeft, numberOfNeuronsInRight, indexOfFirstLeftNeuron, indexOfFirstRightNeuron, indexOfFirstWeight);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            onCudaKernelLaunchFailure("cudaKernel_CalculateLeftLayerDeltas", cudaStatus);
        }
    
        // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            onCudaDeviceSynchronizeError("cudaKernel_CalculateLeftLayerDeltas", cudaStatus);
        }
    }
}//end backpropagationUsingDevice function

/*
 * cudaKernel_CalculateLeftLayerDeltas - calculates and stores the neuron delta for every neuron in left-layer
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * __restrict__ decoration tells NVCC this pointer will only be used to refer to the underlying data (read only)
 * @params: devNeuronDeltas - device copy of float* neuronDeltas
 * @params: devExpected - device copy of float* expected
 * @params: devNeurons - device copy of float* neurons
 * @params: devWeights - device copy of float* weights
 * @params: numberOfNeuronsInLeft - the int number of neurons in left-layer
 * @params: numberOfNeuronsInRight - the int number of neurons in right-layer
 * @params: indexOfFirstLeftNeuron - the int index of the first neuron in left-layer
 * @params: indexOfFirstRightNeuron - the int index of the first neuron in right-layer
 * @params: indexOfFirstWeight - the int index of the first weight between left and right layers
 */
__global__ static void cudaKernel_CalculateLeftLayerDeltas(float* devNeuronDeltas, __restrict__ const float* devExpected, 
                                                           __restrict__ const float* devNeurons, __restrict__ const float* devWeights, 
                                                           const unsigned int numberOfNeuronsInLeft, const unsigned int numberOfNeuronsInRight,
                                                           const unsigned int indexOfFirstLeftNeuron, const unsigned int indexOfFirstRightNeuron, 
                                                           const unsigned int indexOfFirstWeight) {
    volatile unsigned int nl = threadIdx.x + blockIdx.x * blockDim.x; // calculate the thread id (used as offset from indexOfFirstLeftNeuron)

    // check that this thread is within our desired range (extra threads may have been launched for GPU optimization)
    if (nl < numberOfNeuronsInLeft) {
        float errorSum = 0.0; // store neuron nl's error sum of all right-layer connections

        // for each neuron nr in right-layer
        for (int nr = 0; nr < numberOfNeuronsInRight; nr++) {
            // add right-layer neuron nr's delta and it's weighted link to left-layer neuron nl to nl's error sum
            errorSum += ((devNeuronDeltas[indexOfFirstRightNeuron + nr]) * devWeights[indexOfFirstWeight + nr*numberOfNeuronsInLeft]);
        }

        // calculate nl's delta and store in devNeuronDeltas[nl's index]
        // note: we add whatever is in devNeuronDeltas[nl's index] to the delta value (this is so we have a summation of all deltas in a given epoch)
        devNeuronDeltas[indexOfFirstLeftNeuron + nl] += errorSum * sigmoidDerivative(devNeurons[indexOfFirstLeftNeuron + nl]);
    }
}//end cudaKernel_CalculateLeftLayerDeltas function

/*
 * cudaKernel_CalculateOutputLayerDeltas - calculates and stores the neuron delta for every neuron in output-layer
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * __restrict__ decoration tells NVCC this pointer will only be used to refer to the underlying data (read only)
 * @params: devNeuronDeltas - device copy of float* neuronDeltas
 * @params: devExpected - device copy of float* expected
 * @params: devNeurons - device copy of float* neurons
 * @params: numberOfNeuronsInOutput - the int number of neurons in output-layer
 * @params: indexOfFirstOutputNeuron - the int index of the first neuron in output-layer
 */
__global__ static void cudaKernel_CalculateOutputLayerDeltas(float* devNeuronDeltas, __restrict__ const float* devExpected, 
                                                             __restrict__ const float* devNeurons, 
                                                             const unsigned int numberOfNeuronsInOutput, 
                                                             const unsigned int indexOfFirstOutputNeuron) {
    volatile unsigned int no = threadIdx.x + blockIdx.x * blockDim.x; // calculate the thread id (used as offset from indexOfFirstOutputNeuron)

    // check that this thread is within our desired range (extra threads may have been launched for GPU optimization)
    if (no < numberOfNeuronsInOutput) {
        // calculate output layer's neuron's delta and store in devNeuronDeltas[neuron's index]
        // note: we add whatever is in neuronDeltas[neuron's index] to the delta value (this is so we have a summation of all deltas in a given epoch)
        devNeuronDeltas[indexOfFirstOutputNeuron + no] += (quadraticCostDerivative(devExpected[no], devNeurons[indexOfFirstOutputNeuron + no]) * sigmoidDerivative(devNeurons[indexOfFirstOutputNeuron + no]));
    }
} //end cudaKernel_CalculateOutputLayerDeltas function

/*
 * cudaKernel_updateBiases - uses neuronDeltas to update every bias in the network to reduce the error rate 
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * __restrict__ decoration tells NVCC this pointer will only be used to refer to the underlying data (read only)
 * @params: devNeuronDeltas - device copy of float* neuronDeltas
 * @params: devBiases - device copy of float* biases
 * @params: numberOfNeuronsTotal - the int number of neurons total in the network
 * @params: learningRate - the float rate at which we want our network to make adjustments
 */
__global__ static void cudaKernel_updateBiases(__restrict__ const float* devNeuronDeltas, float* devBiases, 
                                               const unsigned int numberOfNeuronsTotal, const float learningRate) {
    volatile unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < numberOfNeuronsTotal) {
        devBiases[id] = devBiases[id] + (learningRate * devNeuronDeltas[id]);
    }
} //end cudaKernel_updateBiases function

/*
 * cudaKernel_updateWeightsBetweenLayers - uses neuronDeltas to update the weights between two layers
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * __restrict__ decoration tells NVCC this pointer will only be used to refer to the underlying data (read only)
 * @params: devNeuronDeltas - a float pointer-pointer to the chunk of memory containing the delta values for each neuron
 * @params: devNeurons - a float pointer to the chunk of memory containing the neuron values
 * @params: devWeights - a float pointer-pointer to the chunk of memory containing the weight values
 * @params: numberOfNeuronsInLeft - the int number of neurons in left-layer
 * @params: numberOfNeuronsInRight - the int number of neurons in right-layer
 * @params: numberOfWeightsBetweenLayers - the int number of weights between left and right layers
 * @params: indexOfFirstLeftNeuron - the int index of the first neuron in left-layer
 * @params: indexOfFirstWeight - the int index of the first weight in between left and right layers
 * @params: learningRate - the rate at which we want our network to make adjustments to the weights
 */
__global__ static void cudaKernel_updateWeightsBetweenLayers(__restrict__ const float* devNeuronDeltas, __restrict__ const float* devNeurons, float* devWeights, 
                                                             const unsigned int numberOfNeuronsInLeft, const unsigned int numberOfNeuronsInRight, 
                                                             const unsigned int numberOfWeightsBetweenLayers, const unsigned int indexOfFirstLeftNeuron, 
                                                             const unsigned int indexOfFirstWeight, const float learningRate) {
    if ((blockIdx.x < numberOfNeuronsInLeft) && (threadIdx.x < numberOfWeightsBetweenLayers)) {
        unsigned int weightIndex = indexOfFirstWeight + numberOfNeuronsInRight * blockIdx.x + threadIdx.x;
        unsigned int neuronIndex = numberOfNeuronsInLeft + blockIdx.x;
        devWeights[weightIndex] = devWeights[weightIndex] - (learningRate * devNeuronDeltas[neuronIndex] * devNeurons[neuronIndex]);
    }
} //end cudaKernel_updateWeights function

/*
 * quadraticCostDerivative - a Quadratic Cost derivative function
 * __host__ decoration tells NVCC this function should run on CPU, and be callable from the CPU host
 * __device__ decoration tells NVCC this function should run on GPU, and be callable from the GPU device
 * @params: expectedValue - a pointer to a float value
 * @params: calculatedValue - a pointer to a float value
 * @returns: the difference between outputExpected and calculated values
 */
__host__ __device__ float quadraticCostDerivative(const float expectedValue, const float calculatedValue) {
    return expectedValue - calculatedValue;
} //end quadraticCostDerivative function

/*
 * updateBiasesUsingDevice - uses devNeuronDeltas to update the devBiases to reduce the error rate using GPU
 * @params: devNeuronDeltas - device copy of float* neuronDeltas
 * @params: devBiases - device copy of float* biases
 * @params: numberOfNeuronsTotal - the int number of neurons total in the network
 * @params: learningRate - the float rate at which we want our network to make adjustments
 */
void updateBiasesUsingDevice(float* devNeuronDeltas, float* devBiases, 
                             const unsigned int numberOfNeuronsTotal, const float learningRate) {
    // use getDeviceProperties helper function to get GPU device information
    unsigned int numberOfSMs = 0; // the number of SMs on the device (1 SM can process 1 block at a time)
    unsigned int warpsize = 0; // the number of threads that an SM can manage at one time
    getDeviceProperties(&numberOfSMs, &warpsize); 

    // set blocks and threads to a size that will fully utilize the GPU (overkill, I know, but we're going for performance here)
    unsigned int blocks = numberOfSMs; // should be equal to the number of SMs on the GPU device after getDeviceProperties
    unsigned int threads = warpsize; // should be equal to the warpsize on the GPU device after getDeviceProperties
    
    // double or devide the number of threads until we have a number close to the number of neurons in output-layer
    threads = getOptimalThreadSize(blocks, threads, numberOfNeuronsTotal, warpsize);

    // use the devNeuronDeltas to update every bias in the network
    cudaKernel_updateBiases<<<blocks, threads>>>(devNeuronDeltas, devBiases, numberOfNeuronsTotal, learningRate);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        onCudaKernelLaunchFailure("cudaKernel_updateBiases", cudaStatus);
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        onCudaDeviceSynchronizeError("cudaKernel_updateBiases", cudaStatus);
    }
}//end updateBiasesUsingDevice function

/*
 * updateBiasesUsingHost - uses neuronDeltas to update the biases to reduce the error rate using CPU
 * @params: neuronDeltas - a float pointer-pointer to the chunk of memory containing the delta values for each neuron
 * @params: biases - a float pointer-pointer to the chunk of memory containing the bias values
 * @params: numberOfNeuronsTotal - the number of total neurons in the network
 * @params: learningRate - the rate at which we want our network to make adjustments to the weights
 */
void updateBiasesUsingHost(const float* neuronDeltas, float** biases, 
                           const unsigned int numberOfNeuronsTotal, const float learningRate) {
    for (int i = 0; i < numberOfNeuronsTotal; i++) {
        (*biases)[i] = (*biases)[i] + (learningRate * neuronDeltas[i]);
    }
} //end updateBiasesUsingHost function

/*
 * updateWeightsUsingDevice - uses devNeuronDeltas to update the devWeights to reduce the error rate using GPU
 * @params: devNeuronDeltas - a float pointer-pointer to the chunk of memory containing the delta values for each neuron
 * @params: devNeurons - a float pointer to the chunk of memory containing the neuron values
 * @params: devWeights - a float pointer-pointer to the chunk of memory containing the weight values
 * @params: numberOfLayers - the int number of layers in the network (index 0 => input, index numberOfLayers-1 => output)
 * @params: numberOfNeuronsInLayer - an int pointer to a chunk of memory containing the number of neurons in each layer
 * @params: numberOfWeightsInFrontOfLayer - an int pointer to a chunk of memory containing the number of weights in front of each layer
 * @params: indexOfFirstNeuronInLayer - an int pointer to a chunk of memory containing the indexes of the first neurons in each layer
 * @params: indexOfFirstWeightInFrontOfLayer - an int pointer to a chunk of memory containing the indexes of the first weight in front of each layer
 * @params: learningRate - the rate at which we want our network to make adjustments to the weights
 */
void updateWeightsUsingDevice(float* devNeuronDeltas, float* devNeurons, float* devWeights, const unsigned int numberOfLayers, 
                              const unsigned int* numberOfNeuronsInLayer, const unsigned int* numberOfWeightsInFrontOfLayer, 
                              const unsigned int* indexOfFirstNeuronInLayer, const unsigned int* indexOfFirstWeightInFrontOfLayer, 
                              const float learningRate) {
    // use getDeviceProperties helper function to get GPU device information
    unsigned int numberOfSMs = 0; // the number of SMs on the device (1 SM can process 1 block at a time)
    unsigned int warpsize = 0; // the number of threads that an SM can manage at one time
    getDeviceProperties(&numberOfSMs, &warpsize); 

    // set blocks and threads to a size that will fully utilize the GPU (overkill, I know, but we're going for performance here)
    unsigned int blocks = numberOfSMs; // should be equal to the number of SMs on the GPU device after getDeviceProperties
    unsigned int threads = warpsize; // should be equal to the warpsize on the GPU device after getDeviceProperties
    
    for (int l = 1; l < numberOfLayers; l++) {
        unsigned int numberOfNeuronsInLeft = numberOfNeuronsInLayer[l-1]; // left-layer size
        unsigned int numberOfNeuronsInRight = numberOfNeuronsInLayer[l]; // right-layer size
        unsigned int numberOfWeightsBetween = numberOfWeightsInFrontOfLayer[l]; // number of weights between left and right layers

        unsigned int indexOfFirstLeftNeuron = indexOfFirstNeuronInLayer[l-1]; // start at this neuron for left-layer
        unsigned int indexOfFirstWeight = indexOfFirstWeightInFrontOfLayer[l]; // start at this weight 

        // double or devide the number of threads until we have a number close to the number of weights between layers
        threads = getOptimalThreadSize(blocks, threads, numberOfWeightsBetween, warpsize);

        // use the devNeuronDeltas to update every bias in the network
        cudaKernel_updateWeightsBetweenLayers<<<blocks, threads>>>(devNeuronDeltas, devNeurons, devWeights, numberOfNeuronsInLeft, numberOfNeuronsInRight, numberOfWeightsBetween, indexOfFirstLeftNeuron, indexOfFirstWeight, learningRate);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            onCudaKernelLaunchFailure("cudaKernel_updateWeightsBetweenLayers", cudaStatus);
        }
    
        // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            onCudaDeviceSynchronizeError("cudaKernel_updateWeightsBetweenLayers", cudaStatus);
        }
    }
}//end updateWeightsUsingDevice function

/*
 * updateWeightsUsingHost - uses neuronDeltas to update the weights to reduce the error rate using CPU
 * @params: neuronDeltas - a float pointer-pointer to the chunk of memory containing the delta values for each neuron
 * @params: neurons - a float pointer to the chunk of memory containing the neuron values
 * @params: weights - a float pointer-pointer to the chunk of memory containing the weight values
 * @params: numberOfLayers - the int number of layers in the network (index 0 => input, index numberOfLayers-1 => output)
 * @params: numberOfNeuronsInLayer - an int pointer to a chunk of memory containing the number of neurons in each layer
 * @params: numberOfWeightsInFrontOfLayer - an int pointer to a chunk of memory containing the number of weights in front of each layer
 * @params: indexOfFirstNeuronInLayer - an int pointer to a chunk of memory containing the indexes of the first neurons in each layer
 * @params: indexOfFirstWeightInFrontOfLayer - an int pointer to a chunk of memory containing the indexes of the first weight in front of each layer
 * @params: learningRate - the rate at which we want our network to make adjustments to the weights
 */
void updateWeightsUsingHost(const float* neuronDeltas, const float* neurons, float** weights, const unsigned int numberOfLayers, 
                            const unsigned int* numberOfNeuronsInLayer, const unsigned int* numberOfWeightsInFrontOfLayer, 
                            const unsigned int* indexOfFirstNeuronInLayer, const unsigned int* indexOfFirstWeightInFrontOfLayer, 
                            const float learningRate) {
    // for each layer l after input layer, update the weights in the layer
    for (int l = 1; l < numberOfLayers; l++) {
        unsigned int numberOfNeuronsInLeft = numberOfNeuronsInLayer[l-1]; // left-layer size
        unsigned int numberOfNeuronsInRight = numberOfNeuronsInLayer[l]; // right-layer size

        unsigned int indexOfFirstLeftNeuron = indexOfFirstNeuronInLayer[l-1]; // start at this neuron for left-layer
        unsigned int indexOfFirstRightNeuron = indexOfFirstNeuronInLayer[l]; // start at this neuron for right-layer
        unsigned int indexOfFirstWeight = indexOfFirstWeightInFrontOfLayer[l]; // start at this weight 

        // for each neuron n in right-layer
        for (int n = 0; n < numberOfNeuronsInRight; n++) {
            unsigned int neuronIndex = indexOfFirstRightNeuron + n; 
            for (int w = 0; w < numberOfNeuronsInLeft; w++) {
                int weightIndex = indexOfFirstWeight + numberOfNeuronsInLeft * n + w;
                (*weights)[weightIndex] = (*weights)[weightIndex] + (learningRate * neuronDeltas[neuronIndex] * neurons[neuronIndex]);
            }
        }
    }
} //end updateWeightsUsingHost function
