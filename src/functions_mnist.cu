/*******************************************************************************************
 * Filename: mnistfunctions.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the functions needed to work with the MNIST data sets.
 *******************************************************************************************
 */

/*
 * getCalculatedMNISTClassificationUsingHost - returns the MNIST class predicted by the neural network
 * @params: neurons - a float pointer-pointer to the chunk of memory containing the neuron values 
 * @params: indexOfFirstOutputNeuron - the int index of the first neuron in output-layer
 * @return: an int number between 0 - 9 (inclusive), the network's MNIST digit prediction
 */
int getCalculatedMNISTClassificationUsingHost(float* neurons, unsigned int indexOfFirstOutputNeuron) {
    float highestValue = 0.0;
    int classification = 0;

    for (int i = 0; i < MNISTCLASSIFICATIONS; i++) {
        if (neurons[indexOfFirstOutputNeuron + i] > highestValue) {
            highestValue = neurons[indexOfFirstOutputNeuron + i];
            classification = i;
        }
    }
    return classification;
} //end getCalculatedMNISTClassificationUsingHost function

/*
 * getCalculatedMNISTClassificationUsingDevice - determines device-calculated classification and puts in devClassification
 * @params: devClassification - device variable to store the classification (will get copied over to host during evaluation)
 * @params: devNeurons - device copy of neurons
 * @params: indexOfFirstOutputNeuron - the int index of the first neuron in output-layer
 */
void getCalculatedMNISTClassificationUsingDevice(int* devClassification, float* devNeurons, unsigned int indexOfFirstOutputNeuron) {
    // determine the MNIST classification calculated by the device and put in devClassification to be extracted later
    cudaKernel_GetCalculatedMNISTClassification<<<1, 1>>>(devClassification, devNeurons, indexOfFirstOutputNeuron);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        onCudaKernelLaunchFailure("cudaKernel_GetCalculatedMNISTClassification", cudaStatus)
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        onCudaDeviceSynchronizeError("cudaKernel_GetCalculatedMNISTClassification", cudaStatus);
    }
} //end getCalculatedMNISTClassificationUsingDevice function

/*
 * loadMnistSampleUsingHost - loads the next sample's data into the network and label into expected
 * @params: mnistLabels - a char pointer to the array of mnist labels for all samples
 * @params: mnistData - an unsigned char pointer to the array of mnist pixel data for all samples
 * @params: indexOfSampleLabel - the int index of the sample label to load
 * @params: indexOfSampleFirstData - the int index of the sample's first data value
 * @params: expected - a float pointer-pointer to the expected output neuron values
 * @params: neurons - a float pointer-pointer to the neuron values
 */
void loadMnistSampleUsingHost(const char* mnistLabels, const unsigned char* mnistData, 
                                  unsigned int indexOfSampleLabel, unsigned int indexOfSampleFirstData, 
                                  float** expected, float** neurons) {
    // load the next sample's label into expected
    for (int i = 0; i < MNISTCLASSIFICATIONS; i++) {
        if (mnistLabels[indexOfSampleLabel] == i) {
            (*expected)[i] = 1.0; // set the expected output for neuron i to 1
        } else {
            (*expected)[i] = 0.0; // set the expected output for neuron i to 0
        }
    }

    // load the next sample's data into network's input-layer
    for (int i = 0; i < MNISTSAMPLEDATASIZE; i++) {
        (*neurons)[i] = (float) mnistData[indexOfSampleFirstData + i];
    }
} //end loadMnistSampleUsingHost function

/*
 * loadMnistSampleUsingDevice - loads the next sample's data into the network and label into devNxpected
 * @params: devMnistLabels - device copy of const char* mnistLabels
 * @params: devMnistData - device copy of const unsigned char* mnistData
 * @params: indexOfSampleLabel - the int index of the sample label to load
 * @params: indexOfSampleFirstData - the int index of the sample's first data value
 * @params: devExpected - device copy of float** expected
 * @params: devNeurons - device copy of float** neurons
 */
void loadMnistSampleUsingDevice(const char* devMnistLabels, const unsigned char* devMnistData, 
                                  unsigned int indexOfSampleLabel, unsigned int indexOfSampleFirstData, 
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

    // load the next sample's label into expected (only 1 block of 32 threads should be needed for the output layer)
    cudaKernel_loadMnistSampleLabelIntoExpected<<<1, threads>>>(devMnistLabels, indexOfSampleLabel, devExpected);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        onCudaKernelLaunchFailure("cudaKernel_loadMnistSampleLabelIntoExpected", cudaStatus)
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        onCudaDeviceSynchronizeError("cudaKernel_loadMnistSampleLabelIntoExpected", cudaStatus);
    }

    // double or devide the number of threads until we have a number close to the number of pixels in each MNIST sample
    threads = getOptimalThreadSize(blocks, threads, MNISTSAMPLEDATASIZE, warpsize);


    // load the next sample's data into network's input-layer
    cudaKernel_loadMnistSampleDataIntoInputLayer<<<blocks, threads>>>(devMnistData, indexOfSampleFirstData, devNeurons);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        onCudaKernelLaunchFailure("cudaKernel_loadMnistSampleDataIntoInputLayer", cudaStatus)
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        onCudaDeviceSynchronizeError("cudaKernel_loadMnistSampleDataIntoInputLayer", cudaStatus);
    }
}//end loadMnistSampleUsingDevice function

/*
 * readMnistTestSamplesFromDisk
 * @params: testData - a pointer to an array of unsigned chars (bytes) containing the pixel values for all test samples
 * @params: testLabels - a pointer to an array of chars (bytes) containing the labels for all test samples
 * @params: numberOfSamples - a pointer to an int value (the number of test samples)
 */
void readMnistTestSamplesFromDisk(unsigned char** testData, char** testLabels, unsigned int* numberOfSamples) {
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
} //end loadMnistTestSamplesFromDisk function

/*
 * readMnistTrainingSamplesFromDisk
 * @params: trainingData - a pointer to an array of unsigned chars (bytes) containing the pixel values for all training samples
 * @params: trainingLabels - a pointer to an array of chars (bytes) containing the labels for all training samples
 * @params: numberOfSamples - a pointer to an int value (the number of training samples)
 */
void readMnistTrainingSamplesFromDisk(unsigned char** trainingData, char** trainingLabels, unsigned int* numberOfSamples) {
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
} //end readMnistTrainingSamplesFromDisk function

/*
 * cudaKernel_GetCalculatedMNISTClassification - ONLY LAUNCH 1 BLOCK WITH 1 THREAD WHEN USING THIS CUDAKERNEL
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devMnistData - device copy of const unsigned char* mnistData
 * @params: indexOfNextSampleFirstData - the int index of the next sample's first data value
 * @params: devNeurons - device copy of float** neurons
 */
__global__ void cudaKernel_GetCalculatedMNISTClassification(int* devClassification, float* devNeurons, unsigned int indexOfFirstOutputNeuron) {
    volatile unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < 1) {
        float highestValue = 0.0;
        devClassification[0] = 0;
        for (int i = 0; i < MNISTCLASSIFICATIONS; i++) {
            if (devNeurons[indexOfFirstOutputNeuron + i] > highestValue) {
                highestValue = devNeurons[indexOfFirstOutputNeuron + i];
                devClassification[0] = i;
            }
        }
    }
} //end cudaKernel_GetCalculatedMNISTClassification function

/*
 * cudaKernel_loadMnistSampleLabelIntoExpected
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devMnistLabels - device copy of const char* mnistLabels
 * @params: indexOfNextSampleLabel - the int index of the next sample label to load
 * @params: devExpected - device copy of float** expected
 */
__global__ void cudaKernel_loadMnistSampleLabelIntoExpected(const char* devMnistLabels, unsigned int indexOfNextSampleLabel, float* devExpected) {
    volatile unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < MNISTCLASSIFICATIONS) {
        if (devMnistLabels[indexOfNextSampleLabel] == id) {
            devOutputExpected[id] = 1;
        } else {
            devOutputExpected[id] = 0;
        }
    }
} //end cudaKernel_loadMnistSampleLabelIntoExpected function

/*
 * cudaKernel_loadMnistSampleDataIntoInputLayer
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devMnistData - device copy of const unsigned char* mnistData
 * @params: indexOfNextSampleFirstData - the int index of the next sample's first data value
 * @params: devNeurons - device copy of float** neurons
 */
__global__ void cudaKernel_loadMnistSampleDataIntoInputLayer(const unsigned char* devMnistData, unsigned int indexOfNextSampleFirstData, float* devNeurons) {
    volatile unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < MNISTSAMPLEDATASIZE) {
        devNeurons[id] = (float) devMnistData[indexOfNextSampleFirstData + id];
    }
} //end cudaKernel_loadMnistSampleDataIntoInputLayer function

