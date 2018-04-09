/*******************************************************************************************
 * Filename: mnistfunctions.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the functions needed to work with the MNIST data sets.
 *******************************************************************************************
 */

/*
 * getCalculatedMNISTClassification - returns the MNIST class predicted by the neural network
 * @params: neurons - a float pointer-pointer to the chunk of memory containing the neuron values 
 * @params: indexOfFirstOutputNeuron - 
 * @return: an int number between 0 - 9 (inclusive), the network's MNIST digit prediction
 */
int getCalculatedMNISTClassification(float* neurons, unsigned int indexOfFirstOutputNeuron) {
    float highestValue = 0.0;
    int classification = 0;

    for (int i = 0; i < MNISTCLASSIFICATIONS; i++) {
        if (neurons[indexOfFirstOutputNeuron + i] > highestValue) {
            highestValue = neurons[indexOfFirstOutputNeuron + i];
            classification = i;
        }
    }
    return classification;
} //end getCalculatedMNISTClassification function

/*
 * loadMnistTestSamples
 * @params: testData - a pointer to an array of unsigned chars (bytes) containing the pixel values for all test samples
 * @params: testLabels - a pointer to an array of chars (bytes) containing the labels for all test samples
 * @params: numberOfSamples - a pointer to an int value (the number of test samples)
 */
void loadMnistTestSamples(unsigned char** testData, char** testLabels, unsigned int* numberOfSamples) {
    FILE* thefile = fopen(MNISTTESTFILELOCATION, "r");
    if (thefile == NULL) {
        onFileOpenError (MNISTTESTFILELOCATION);
    }

    *numberOfSamples = MNISTTESTSETSIZE;

    // stretch trainingData array to be MNISTSAMPLEDATASIZE * MNISTTRAININGSETSIZE * sizeof(char)
    unsigned char* tempPtr = (*testData); // keep track of old memory
    (*testData) = (unsigned char*) malloc(MNISTSAMPLEDATASIZE * MNISTTESTSETSIZE * sizeof(unsigned char));
    if ((*testData) == NULL) {
        onMallocError(MNISTSAMPLEDATASIZE * MNISTTESTSETSIZE * sizeof(char));
    }
    free(tempPtr); // release old memory

    // stretch trainingLabels array to be MNISTTRAININGSETSIZE * sizeof(char)
    char* tempPtr2 = (*testLabels); // keep track of old memory
    (*testLabels) = (char*) malloc(MNISTTESTSETSIZE * sizeof(char));
    if ((*testLabels) == NULL) {
        onMallocError(MNISTTESTSETSIZE * sizeof(char));
    }
    free(tempPtr2); // release old memory

    // setup variables needed for getline function
    char* buffer;
    size_t bufsize = 785 * 4; // for all 785 entries: 3 chars for the number, 1 char for the comma,
    size_t characters;

    buffer = (char*) malloc(bufsize * sizeof(char));
    if (buffer == NULL) {
        onMallocError(bufsize * sizeof(char));
    }

    unsigned int sampleIndex = 0;
    unsigned int dataIndex = 0;

    // loop through thefile, one line (one sample) at a time
    while ((characters = getline(&buffer, &bufsize, thefile)) != -1) {
        char* token = strtok(buffer, VALUEDELIM); // grab first token from the line (the sample's label)
        sscanf(token, "%hhu", &((*testLabels)[sampleIndex])); // store the sample's label
        token = strtok(NULL, VALUEDELIM); // grab next token from line (the first pixel value)
        while (token) {
            sscanf(token, "%hhu", &((*testData)[dataIndex])); // store sample's data
            token = strtok(NULL, VALUEDELIM); // grab next token from line
            dataIndex++;
        }
        sampleIndex++; // prepare to store next sample label and data
    }

    free(buffer);
} //end loadMnistTestSamples function

/*
 * loadMnistTrainingSamples
 * @params: trainingData - a pointer to an array of unsigned chars (bytes) containing the pixel values for all training samples
 * @params: trainingLabels - a pointer to an array of chars (bytes) containing the labels for all training samples
 * @params: numberOfSamples - a pointer to an int value (the number of training samples)
 */
void loadMnistTrainingSamples(unsigned char** trainingData, char** trainingLabels, unsigned int* numberOfSamples) {
    FILE* thefile = fopen(MNISTTRAINFILELOCATION, "r");
    if (thefile == NULL) {
        onFileOpenError (MNISTTRAINFILELOCATION);
    }

    *numberOfSamples = MNISTTRAININGSETSIZE;

    // stretch trainingData array to be MNISTSAMPLEDATASIZE * MNISTTRAININGSETSIZE * sizeof(char)
    unsigned char* tempPtr = (*trainingData); // keep track of old memory
    (*trainingData) = (unsigned char*) malloc(MNISTSAMPLEDATASIZE * MNISTTRAININGSETSIZE * sizeof(unsigned char));
    if ((*trainingData) == NULL) {
        onMallocError(MNISTSAMPLEDATASIZE * MNISTTRAININGSETSIZE * sizeof(char));
    }
    free(tempPtr); // release old memory

    // stretch trainingLabels array to be MNISTTRAININGSETSIZE * sizeof(char)
    char* tempPtr2 = (*trainingLabels); // keep track of old memory
    (*trainingLabels) = (char*) malloc(MNISTTRAININGSETSIZE * sizeof(char));
    if ((*trainingLabels) == NULL) {
        onMallocError(MNISTTRAININGSETSIZE * sizeof(char));
    }
    free(tempPtr2); // release old memory

    // setup variables needed for getline function
    char* buffer;
    size_t bufsize = 785 * 4; // for all 785 entries: 3 chars for the number, 1 char for the comma,
    size_t characters;

    buffer = (char*) malloc(bufsize * sizeof(char));
    if (buffer == NULL) {
        onMallocError(bufsize * sizeof(char));
    }

    unsigned int sampleIndex = 0;
    unsigned int dataIndex = 0;

    // loop through thefile, one line (one sample) at a time
    while ((characters = getline(&buffer, &bufsize, thefile)) != -1) {
        char* token = strtok(buffer, VALUEDELIM); // grab first token from the line (the sample's label)
        sscanf(token, "%hhu", &((*trainingLabels)[sampleIndex])); // store the sample's label
        token = strtok(NULL, VALUEDELIM); // grab next token from line (the first pixel value)
        while (token) {
            sscanf(token, "%hhu", &((*trainingData)[dataIndex])); // store sample's data
            token = strtok(NULL, VALUEDELIM); // grab next token from line
            dataIndex++;
        }
        sampleIndex++; // prepare to store next sample label and data
    }

    free(buffer);
} //end loadMnistTrainingSamples function

/*
 * loadNextMnistSampleUsingHost - loads the next sample's data into the network and label into expected
 * @params: mnistLabels - a char pointer to the array of mnist labels for all samples
 * @params: mnistData - an unsigned char pointer to the array of mnist pixel data for all samples
 * @params: indexOfNextSampleLabel - the int index of the next sample label to load
 * @params: indexOfNextSampleFirstData - the int index of the next sample's first data value
 * @params: expected - a float pointer-pointer to the expected output neuron values
 * @params: neurons - a float pointer-pointer to the neuron values
 */
void loadNextMnistSampleUsingHost(const char* mnistLabels, const unsigned char* mnistData, 
                                  unsigned int indexOfNextSampleLabel, unsigned int indexOfNextSampleFirstData, 
                                  float** expected, float** neurons) {
    // load the next sample's label into expected
    for (int i = 0; i < MNISTCLASSIFICATIONS; i++) {
        if (mnistLabels[indexOfNextSampleLabel] == i) {
            (*expected)[i] = 1.0; // set the expected output for neuron i to 1
        } else {
            (*expected)[i] = 0.0; // set the expected output for neuron i to 0
        }
    }

    // load the next sample's data into network's input-layer
    for (int i = 0; i < MNISTSAMPLEDATASIZE; i++) {
        (*neurons)[i] = (float) mnistData[indexOfNextSampleFirstData + i];
    }
} //end loadNextMnistSampleUsingHost function

/*
 * loadNextMnistSampleUsingDevice - loads the next sample's data into the network and label into devNxpected
 * @params: devMnistLabels - device copy of const char* mnistLabels
 * @params: devMnistData - device copy of const unsigned char* mnistData
 * @params: indexOfNextSampleLabel - the int index of the next sample label to load
 * @params: indexOfNextSampleFirstData - the int index of the next sample's first data value
 * @params: devExpected - device copy of float** expected
 * @params: devNeurons - device copy of float** neurons
 */
void loadNextMnistSampleUsingDevice(const char* devMnistLabels, const unsigned char* devMnistData, 
                                  unsigned int indexOfNextSampleLabel, unsigned int indexOfNextSampleFirstData, 
                                  float* devExpected, float* devNeurons) {
    // use getDeviceProperties helper function to get GPU device information
    unsigned int numberOfSMs = 0; // the number of SMs on the device (1 SM can process 1 block at a time)
    unsigned int warpsize = 0; // the number of threads that an SM can manage at one time
    getDeviceProperties(&numberOfSMs, &warpsize); 

    // set blocks and threads to a size that will fully utilize the GPU (overkill, I know, but we're going for performance here)
    unsigned int blocks = numberOfSMs; // should be equal to the number of SMs on the GPU device after getDeviceProperties
    unsigned int threads = warpsize; // should be equal to the warpsize on the GPU device after getDeviceProperties
    
    // double or devide the number of threads until we have a number close to the number of possible MNIST classifications
    threads = getOptimalThreadSize(blocks, threads, MNISTCLASSIFICATIONS, warpsize);

    // load the next sample's label into expected
    cudaKernel_loadNextMnistSampleLabelIntoExpected<<<blocks, threads>>>(devMnistLabels, indexOfNextSampleLabel, devExpected);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        onCudaKernelLaunchFailure("cudaKernel_loadNextMnistSampleLabelIntoExpected", cudaStatus)
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        onCudaDeviceSynchronizeError("cudaKernel_loadNextMnistSampleLabelIntoExpected", cudaStatus);
    }

    // double or devide the number of threads until we have a number close to the number of pixels in each MNIST sample
    threads = getOptimalThreadSize(blocks, threads, MNISTSAMPLEDATASIZE, warpsize);


    // load the next sample's data into network's input-layer
    cudaKernel_loadNextMnistSampleDataIntoInputLayer<<<blocks, threads>>>(devMnistData, indexOfNextSampleFirstData, devNeurons);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        onCudaKernelLaunchFailure("cudaKernel_loadNextMnistSampleDataIntoInputLayer", cudaStatus)
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        onCudaDeviceSynchronizeError("cudaKernel_loadNextMnistSampleDataIntoInputLayer", cudaStatus);
    }
}//end loadNextMnistSampleUsingDevice function

/*
 * cudaKernel_loadNextMnistSampleLabelIntoExpected
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devMnistLabels - device copy of const char* mnistLabels
 * @params: indexOfNextSampleLabel - the int index of the next sample label to load
 * @params: devExpected - device copy of float** expected
 */
__global__ void cudaKernel_loadNextMnistSampleLabelIntoExpected(const char* devMnistLabels, int indexOfNextSampleLabel, float* devExpected) {
    volatile unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < MNISTCLASSIFICATIONS) {
        if (devMnistLabels[indexOfNextSampleLabel] == id) {
            devOutputExpected[id] = 1;
        } else {
            devOutputExpected[id] = 0;
        }
    }
} //end cudaKernel_loadNextMnistSampleLabelIntoExpected function

/*
 * loadNextMnistSampleDataKernel
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devMnistData - device copy of const unsigned char* mnistData
 * @params: indexOfNextSampleFirstData - the int index of the next sample's first data value
 * @params: devNeurons - device copy of float** neurons
 */
__global__ void cudaKernel_loadNextMnistSampleDataIntoInputLayer(const unsigned char* devMnistData, int indexOfNextSampleFirstData, float* devNeurons) {
    volatile unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < MNISTSAMPLEDATASIZE) {
        devNeurons[id] = (float) devMnistData[indexOfNextSampleFirstData + id];
    }
} //end loadNextMnistSampleData kernel function

