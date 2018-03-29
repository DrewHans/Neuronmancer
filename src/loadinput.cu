/*******************************************************************************************
 * Filename: neuralnetworkcore.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the functions needed to load input into the network.
 *******************************************************************************************
 */

/*
 * loadInput
 * @params: neurons - a pointer to an array of double values (the input layer)
 * @params: n - the size of the input layer
 */
void loadInput(double* neurons, int n) {
    // generate random doubles for testing purposes
    for (int i = 0; i < n; i++) {
        neurons[i] = ((double) rand()) / ((double) rand());
    }
} //end loadInput method

/*
 * loadMnistTestSamples
 * @params: testData - a pointer to an array of unsigned chars (bytes) containing the pixel values for all test samples
 * @params: testLabels - a pointer to an array of chars (bytes) containing the labels for all test samples
 * @params: numberOfSamples - a pointer to an int value (the number of test samples)
 */
void loadMnistTestSamples(unsigned char* testData, char* testLabels, int* numberOfSamples) {
    FILE* thefile = fopen(MNISTTESTFILELOCATION, "r");
    if (thefile == NULL) {
        onFileOpenError (MNISTTESTFILELOCATION);
    }

	*numberOfSamples = MNISTTESTSETSIZE;
    
	// stretch trainingData array to be MNISTSAMPLEDATASIZE * MNISTTESTSETSIZE * sizeof(char)
	unsigned char* tempPtr = testData; // keep track of old memory
	testData = (unsigned char*) malloc(MNISTSAMPLEDATASIZE * MNISTTESTSETSIZE * sizeof(char));
    if (trainingData == NULL) {
        onMallocError(MNISTSAMPLEDATASIZE * MNISTTESTSETSIZE * sizeof(char));
    }
    free(tempPtr); // release old memory

    // stretch testLabels array to be MNISTTESTSETSIZE * sizeof(char)
    char* tempPtr2 = testLabels; // keep track of old memory
    testLabels = (char*) malloc(MNISTTESTSETSIZE * sizeof(char));
    if (testLabels == NULL) {
        onMallocError(MNISTTESTSETSIZE * sizeof(char));
    }
    free(tempPtr2); // release old memory
    
    // setup variables needed for getdelim function
    char* buffer = NULL; // stores stuff we pull from thefile
    size_t lineLength; // store the number of chars shoved into buffer (not really needed, but nice to have)
    ssize_t readStatus; // used to detect read error

    // get test label and pixel data for each sample
    for (int i = 0; i < MNISTTESTSETSIZE; i++) {
        // get training label
        readStatus = getdelim(&buffer, &lineLength, VALUEDELIM, thefile);
        if (readStatus == -1) {
            onFileOpenError (MNISTTESTFILELOCATION);
        }
        sscanf(buffer, "%d", testLabels[i]);

    	// get the pixel data
    	for (int j = 0; j < MNISTSAMPLEDATASIZE; j++) {
            readStatus = getdelim(&buffer, &lineLength, VALUEDELIM, thefile);
            if (readStatus == -1) {
                onFileOpenError (MNISTTESTFILELOCATION);
            }
            sscanf(buffer, "%d", testData[j]);
    	}
    	
    	// throw away the newline at the end of the sample entry
    	getdelim(&buffer, &lineLength, "\n", thefile);
    }

}//end loadMnistTestSamples method

/*
 * loadMnistTrainingSamples
 * @params: trainingData - a pointer to an array of unsigned chars (bytes) containing the pixel values for all training samples
 * @params: trainingLabels - a pointer to an array of chars (bytes) containing the labels for all training samples
 * @params: numberOfSamples - a pointer to an int value (the number of training samples)
 */
void loadMnistTrainingSamples(unsigned char* trainingData, char* trainingLabels, int* numberOfSamples) {
    FILE* thefile = fopen(MNISTTRAINFILELOCATION, "r");
    if (thefile == NULL) {
        onFileOpenError (MNISTTRAINFILELOCATION);
    }

	*numberOfSamples = MNISTTRAININGSETSIZE;
    
	// stretch trainingData array to be MNISTSAMPLEDATASIZE * MNISTTRAININGSETSIZE * sizeof(char)
	unsigned char* tempPtr = trainingData; // keep track of old memory
    trainingData = (unsigned char*) malloc(MNISTSAMPLEDATASIZE * MNISTTRAININGSETSIZE * sizeof(char));
    if (trainingData == NULL) {
        onMallocError(MNISTSAMPLEDATASIZE * MNISTTRAININGSETSIZE * sizeof(char));
    }
    free(tempPtr); // release old memory

    // stretch trainingLabels array to be MNISTTRAININGSETSIZE * sizeof(char)
    char* tempPtr2 = trainingLabels; // keep track of old memory
    trainingLabels = (char*) malloc(MNISTTRAININGSETSIZE * sizeof(char));
    if (trainingLabels == NULL) {
        onMallocError(MNISTTRAININGSETSIZE * sizeof(char));
    }
    free(tempPtr2); // release old memory
    
    // setup variables needed for getdelim function
    char* buffer = NULL; // stores stuff we pull from thefile
    size_t lineLength; // store the number of chars shoved into buffer (not really needed, but nice to have)
    ssize_t readStatus; // used to detect read error

    // get training label and pixel data for each sample
    for (int i = 0; i < MNISTTRAININGSETSIZE; i++) {
        // get training label
        readStatus = getdelim(&buffer, &lineLength, VALUEDELIM, thefile);
        if (readStatus == -1) {
            onFileOpenError (MNISTTRAINFILELOCATION);
        }
        sscanf(buffer, "%d", trainingLabels[i]);

    	// get the pixel data
    	for (int j = 0; j < MNISTSAMPLEDATASIZE; j++) {
            readStatus = getdelim(&buffer, &lineLength, VALUEDELIM, thefile);
            if (readStatus == -1) {
                onFileOpenError (MNISTTRAINFILELOCATION);
            }
            sscanf(buffer, "%d", trainingData[j]);
    	}
    	
    	// throw away the newline at the end of the sample entry
    	getdelim(&buffer, &lineLength, "\n", thefile);
    }

}//end loadMnistTrainingSamples method