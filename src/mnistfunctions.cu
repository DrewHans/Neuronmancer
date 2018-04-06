/*******************************************************************************************
 * Filename: mnistfunctions.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the functions needed to work with the MNIST data sets.
 *******************************************************************************************
 */

/*
 * getCalculatedMnistSampleClassification
 * @params: neurons
 * @params: indexStart
 * @return: a number between 0 - 9 (inclusive), the network's digit prediction
 */
int getCalculatedMnistSampleClassification(double* neurons, int indexStart) {
    double highestValue = 0.0;
    int classification = 0;

    for (int i = 0; i < 10; i++) {
        if (neurons[indexStart + i] > highestValue) {
            highestValue = neurons[indexStart + i];
            classification = i;
        }
    }
    return classification;
} //end getCalculatedMnistSampleClassification function

/*
 * loadMnistTestSamples
 * @params: testData - a pointer to an array of unsigned chars (bytes) containing the pixel values for all test samples
 * @params: testLabels - a pointer to an array of chars (bytes) containing the labels for all test samples
 * @params: numberOfSamples - a pointer to an int value (the number of test samples)
 */
void loadMnistTestSamples(unsigned char** testData, char** testLabels, int* numberOfSamples) {
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

    int sampleIndex = 0;
    int dataIndex = 0;

    // loop through thefile, one line (one sample) at a time
    while ((characters = getline(&buffer, &bufsize, thefile)) != -1) {
        char* token = strtok(buffer, ","); // grab first token from the line (the sample's label)
        sscanf(token, "%hhu", &((*testLabels)[sampleIndex])); // store the sample's label
        token = strtok(NULL, ","); // grab next token from line (the first pixel value)
        while (token) {
            sscanf(token, "%hhu", &((*testData)[dataIndex])); // store sample's data
            token = strtok(NULL, ","); // grab next token from line
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
void loadMnistTrainingSamples(unsigned char** trainingData, char** trainingLabels, int* numberOfSamples) {
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

    int sampleIndex = 0;
    int dataIndex = 0;

    // loop through thefile, one line (one sample) at a time
    while ((characters = getline(&buffer, &bufsize, thefile)) != -1) {
        char* token = strtok(buffer, ","); // grab first token from the line (the sample's label)
        sscanf(token, "%hhu", &((*trainingLabels)[sampleIndex])); // store the sample's label
        token = strtok(NULL, ","); // grab next token from line (the first pixel value)
        while (token) {
            sscanf(token, "%hhu", &((*trainingData)[dataIndex])); // store sample's data
            token = strtok(NULL, ","); // grab next token from line
            dataIndex++;
        }
        sampleIndex++; // prepare to store next sample label and data
    }

    free(buffer);
} //end loadMnistTrainingSamples function

/*
 * loadNextMnistSampleData
 * @params: neurons - a pointer to an array of double values (the input layer)
 * @params: mnistData - a pointer to an array of unsigned chars (bytes) containing the pixel values for all mnist samples
 * @params: mnistSampleDataIndexStart - the starting index for the next mnist sample
 */
void loadNextMnistSampleData(double** neurons, const unsigned char* mnistData, int mnistSampleDataIndexStart) {
    for (int i = 0; i < MNISTSAMPLEDATASIZE; i++) {
        (*neurons)[i] = (double) mnistData[mnistSampleDataIndexStart * MNISTSAMPLEDATASIZE + i];
    }
} //end loadNextMnistSampleData function

/*
 * loadNextMnistSampleDataKernel
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devNeurons
 * @params: devMnistData
 * @params: mnistSampleDataIndexStart
 */
__global__ void loadNextMnistSampleDataKernel(double* devNeurons, const unsigned char* devMnistData, int mnistSampleDataIndexStart) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < MNISTSAMPLEDATASIZE) {
        devNeurons[id] = (double) devMnistData[mnistSampleDataIndexStart * MNISTSAMPLEDATASIZE + id];
    }
} //end loadNextMnistSampleData kernel function

/*
 * loadNextMnistSampleLabel
 * @params: outputExpected
 * @params: mnistLabels
 * @params: mnistSampleLabelIndex
 */
void loadNextMnistSampleLabel(double** outputExpected, const char* mnistLabels, int mnistSampleLabelIndex) {
    // for each neuron in the output layer
    for (int i = 0; i < 10; i++) {
        if (mnistLabels[mnistSampleLabelIndex] == i) {
            (*outputExpected)[i] = 1; // set the expected output for neuron i to 1
        } else {
            (*outputExpected)[i] = 0; // set the expected output for neuron i to 0
        }
    }
} //end loadNextMnistSampleLabel function

/*
 * loadNextMnistSampleLabelKernel
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devOutputExpected
 * @params: devMnistLabels
 * @params: mnistSampleLabelIndex
 */
__global__ void loadNextMnistSampleLabelKernel(double* devOutputExpected, const char* devMnistLabels, int mnistSampleLabelIndex) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < 10) {
        if (devMnistLabels[mnistSampleLabelIndex] == id) {
            devOutputExpected[id] = 1;
        } else {
            devOutputExpected[id] = 0;
        }
    }
} //end loadNextMnistSampleLabelKernel kernel function
