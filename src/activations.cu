/*******************************************************************************************
 * Filename: activations.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the host and device activation functions and cudaKernels
 *              for using a single activation function on every neuron in a single layer.
 *******************************************************************************************
 */

/*
 * sigmoidFunction - a Sigmoid activation function
 * __host__ decoration tells NVCC this function should run on CPU, and be callable from the CPU host
 * __device__ decoration tells NVCC this function should run on GPU, and be callable from the GPU device
 * @params: d - a float value
 * @returns: the Sigmoid of d
 */
__host__ __device__ float sigmoidFunction(const float d) {
    return 1.0 / (1.0 + exp(-1.0 * d));
} //end sigmoid activation function

/*
 * sigmoidDerivative - a Sigmoid derivative function
 * __host__ decoration tells NVCC this function should run on CPU, and be callable from the CPU host
 * __device__ decoration tells NVCC this function should run on GPU, and be callable from the GPU device
 * @params: d - a float value
 * @returns: the Sigmoid derivative of d
 */
__host__ __device__ float sigmoidDerivative(const float d) {
    return sigmoidFunction(d) * (1.0 - sigmoidFunction(d));
} //end sigmoid derivative function

/*
 * tanhFunction - a TanH activation function
 * __host__ decoration tells NVCC this function should run on CPU, and be callable from the CPU host
 * __device__ decoration tells NVCC this function should run on GPU, and be callable from the GPU device
 * @params: d - a float value
 * @returns: the TanH of d
 */
__host__ __device__ float tanhFunction(const float d) {
    return (2.0 / (1.0 + exp(-2.0 * d))) - 1.0;
} //end tanh activation function

/*
 * tanhDerivative - a TanH derivative function
 * __host__ decoration tells NVCC this function should run on CPU, and be callable from the CPU host
 * __device__ decoration tells NVCC this function should run on GPU, and be callable from the GPU device
 * @params: d - a float value
 * @returns: the TanH derivative of d
 */
__host__ __device__ float tanhDerivative(const float d) {
    return 1.0 - pow(tanhFunction(d), 2.0);
} //end tanh derivative function

/*
 * reluFunction - a ReLU activation function
 * __host__ decoration tells NVCC this function should run on CPU, and be callable from the CPU host
 * __device__ decoration tells NVCC this function should run on GPU, and be callable from the GPU device
 * @params: d - a float value
 * @returns: the ReLU of d
 */
__host__ __device__ float reluFunction(const float d) {
    if (d < 0) {
        return 0.0;
    } else {
        return d;
    }
} //end relu activation function

/*
 * reluDerivative - a ReLU derivative function
 * __host__ decoration tells NVCC this function should run on CPU, and be callable from the CPU host
 * __device__ decoration tells NVCC this function should run on GPU, and be callable from the GPU device
 * @params: d - a float value
 * @returns: the ReLU derivative of d
 */
__host__ __device__ float reluDerivative(const float d) {
    if (d < 0) {
        return 0.0;
    } else {
        return 1.0;
    }
} //end relu derivative function

/*
 * cudaKernel_ActivateLayerUsingSigmoid - applies Sigmoid activation to every neuron in a layer
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * __restrict__ decoration tells NVCC this function's pointer-parameters are not aliased
 * @params: devNeurons - a pointer to an array of float values in GPU device memory
 * @params: indexOfFirstNeuronInLayer - the index of the first neuron in the layer
 * @params: numberOfNeuronsInLayer - the total number of neurons in the layer
 */
__global__ void cudaKernel_ActivateLayerUsingSigmoid(float* __restrict__ devNeurons, const unsigned int indexOfFirstNeuronInLayer,
        const unsigned int numberOfNeuronsInLayer) {
    volatile unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    // check that this thread is within our desired range (extra threads may have been launched for GPU optimization)
    if (id < numberOfNeuronsInLayer) {
        devNeurons[indexOfFirstNeuronInLayer + id] = sigmoidFunction(devNeurons[indexOfFirstNeuronInLayer + id]);
    }
} //end cudaKernel_ActivateLayerUsingSigmoid function

/*
 * cudaKernel_ActivateLayerUsingTanh - applies TanH activation to every neuron in a layer
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * __restrict__ decoration tells NVCC this function's pointer-parameters are not aliased
 * @params: devNeurons - a pointer to an array of float values in GPU device memory
 * @params: indexOfFirstNeuronInLayer - the index of the first neuron in the layer
 * @params: numberOfNeuronsInLayer - the total number of neurons in the layer
 */
__global__ void cudaKernel_ActivateLayerUsingTanh(float* __restrict__ devNeurons, const unsigned int indexOfFirstNeuronInLayer,
        const unsigned int numberOfNeuronsInLayer) {
    volatile unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    // check that this thread is within our desired range (extra threads may have been launched for GPU optimization)
    if (id < numberOfNeuronsInLayer) {
        devNeurons[indexOfFirstNeuronInLayer + id] = tanhFunction(devNeurons[indexOfFirstNeuronInLayer + id]);
    }
} //end cudaKernel_ActivateLayerUsingTanh function

/*
 * cudaKernel_ActivateLayerUsingRelu - applies ReLU activation to every neuron in a layer
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * __restrict__ decoration tells NVCC this function's pointer-parameters are not aliased
 * @params: devNeurons - a pointer to an array of float values in GPU device memory
 * @params: indexOfFirstNeuronInLayer - the index of the first neuron in the layer
 * @params: numberOfNeuronsInLayer - the total number of neurons in the layer
 */
__global__ void cudaKernel_ActivateLayerUsingRelu(float* __restrict__ devNeurons, const unsigned int indexOfFirstNeuronInLayer,
        const unsigned int numberOfNeuronsInLayer) {
    volatile unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    // check that this thread is within our desired range (extra threads may have been launched for GPU optimization)
    if (id < numberOfNeuronsInLayer) {
        devNeurons[indexOfFirstNeuronInLayer + id] = reluFunction(devNeurons[indexOfFirstNeuronInLayer + id]);
    }
} //end cudaKernel_ActivateLayerUsingRelu function
