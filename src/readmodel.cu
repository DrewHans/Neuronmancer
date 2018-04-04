/*******************************************************************************************
 * Filename: readmodel.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the functions for reading a model from disk.
 *******************************************************************************************
 */

/*
 * readBiasesFromDisk
 * @params: biases - pointer to an array of double values (used to store value before return)
 * @params: numberOfBiasesTotal - equal to numberOfNeuronsTotal
 */
void readBiasesFromDisk(double** biases, int numberOfBiasesTotal) {
    FILE* thefile = fopen(BIASESFILELOCATION, "r");
    if (thefile == NULL) {
        onFileOpenError (BIASESFILELOCATION);
    }

    // stretch array to be numberOfWeightsTotal * sizeof(double)
    double* tempPtr = (*biases); // keep track of old memory
    (*biases) = (double*) malloc(numberOfBiasesTotal * sizeof(double));
    if ((*biases) == NULL) {
        onMallocError(numberOfBiasesTotal * sizeof(double));
    }
    free(tempPtr); // release old memory

    // setup variables needed for getdelim function
    char* buffer = NULL; // stores stuff we pull from thefile
    size_t lineLength; // store the number of chars shoved into buffer (not really needed, but nice to have)
    ssize_t readStatus; // used to detect read error

    for (int i = 0; i < numberOfBiasesTotal; i++) {
        // get biases[i] from file
        readStatus = getdelim(&buffer, &lineLength, VALUEDELIM, thefile);
        if (readStatus == -1) {
            onFileOpenError (BIASESFILELOCATION);
        }
        sscanf(buffer, "%lf", &((*biases)[i])); // convert buffer string to double and shove in biases[i]
    }

    fclose(thefile); // close the file once we're done with it
} //end readBiasesFromDisk function

/*
 * readEpochsFromDisk
 * @params: epochs - pointer to an int value (used to store value before return)
 */
void readEpochsFromDisk(int* epochs) {
    FILE* thefile = fopen(EPOCHSFILELOCATION, "r");
    if (thefile == NULL) {
        onFileOpenError (EPOCHSFILELOCATION);
    }
    // setup variables needed for getdelim function
    char* buffer = NULL; // stores stuff we pull from thefile
    size_t lineLength; // store the number of chars shoved into buffer (not really needed, but nice to have)
    ssize_t readStatus; // used to detect read error

    // get learningRate from file
    readStatus = getdelim(&buffer, &lineLength, '\n', thefile);
    if (readStatus == -1) {
        onFileOpenError (EPOCHSFILELOCATION);
    }
    sscanf(buffer, "%d", epochs); // convert buffer string to double and shove in epochs

    fclose(thefile); // close the file once we're done with it
} //end readEpochsFromDisk function

/*
 * readLearningRateFromDisk
 * @params: learningRate - pointer to a double value (used to store value before return)
 */
void readLearningRateFromDisk(double* learningRate) {
    FILE* thefile = fopen(LEARNINGRATEFILELOCATION, "r");
    if (thefile == NULL) {
        onFileOpenError (LEARNINGRATEFILELOCATION);
    }

    // setup variables needed for getdelim function
    char* buffer = NULL; // stores stuff we pull from thefile
    size_t lineLength; // store the number of chars shoved into buffer (not really needed, but nice to have)
    ssize_t readStatus; // used to detect read error

    // get learningRate from file
    readStatus = getdelim(&buffer, &lineLength, '\n', thefile);
    if (readStatus == -1) {
        onFileOpenError (LEARNINGRATEFILELOCATION);
    }
    sscanf(buffer, "%lf", learningRate); // convert buffer string to double and shove in learningRate

    fclose(thefile); // close the file once we're done with it
} //end readLearningRateFromDisk function

/*
 * readWeightsFromDisk
 * @params: weights - pointer to an array of double values (used to store value before return)
 * @params: numberOfWeightsTotal - pointer to an int value
 */
void readWeightsFromDisk(double** weights, int numberOfWeightsTotal) {
    FILE* thefile = fopen(WEIGHTSFILELOCATION, "r");
    if (thefile == NULL) {
        onFileOpenError (WEIGHTSFILELOCATION);
    }

    // stretch array to be numberOfWeightsTotal * sizeof(double)
    double* tempPtr = (*weights); // keep track of old memory
    (*weights) = (double*) malloc(numberOfWeightsTotal * sizeof(double));
    if ((*weights) == NULL) {
        onMallocError(numberOfWeightsTotal * sizeof(double));
    }
    free(tempPtr); // release old memory

    // setup variables needed for getdelim function
    char* buffer = NULL; // stores stuff we pull from thefile
    size_t lineLength; // store the number of chars shoved into buffer (not really needed, but nice to have)
    ssize_t readStatus; // used to detect read error

    for (int i = 0; i < numberOfWeightsTotal; i++) {
        // get weights[i] from file
        readStatus = getdelim(&buffer, &lineLength, VALUEDELIM, thefile);
        if (readStatus == -1) {
            onFileOpenError (WEIGHTSFILELOCATION);
        }
        sscanf(buffer, "%lf", &((*weights)[i])); // convert buffer string to double and shove in weights[i]
    }

    fclose(thefile); // close the file once we're done with it
} //end readWeightsFromDisk function

/*
 * readModelValuesFromDisk
 * @params: numberOfLayers - pointer to an int value (used to store value before return)
 * @params: numberOfNeuronsTotal - pointer to an int value (used to store value before return)
 * @params: numberOfWeightsTotal - pointer to an int value (used to store value before return)
 * @params: numberOfNeuronsPerLayer - pointer to an array of int values (used to store value before return)
 * @params: numberOfWeightsPerLayer - pointer to an array of int values (used to store value before return)
 * @params: firstNeuronIndexPerLayer - pointer to an array of int values (used to store value before return)
 * @params: firstWeightIndexPerLayer - pointer to an array of int values (used to store value before return)
 */
void readModelValuesFromDisk(int* numberOfLayers, int* numberOfNeuronsTotal, int* numberOfWeightsTotal, int** numberOfNeuronsPerLayer,
        int** numberOfWeightsPerLayer, int** firstNeuronIndexPerLayer, int** firstWeightIndexPerLayer) {
    FILE* thefile = fopen(MODELVALUESLOCATION, "r");
    if (thefile == NULL) {
        onFileOpenError (MODELVALUESLOCATION);
    }

    // setup variables needed for getdelim function
    char* buffer = NULL; // stores stuff we pull from thefile
    size_t lineLength; // store the number of chars shoved into buffer (not really needed, but nice to have)
    ssize_t readStatus; // used to detect read error

    // get numberOfLayers from file
    readStatus = getdelim(&buffer, &lineLength, VALUEDELIM, thefile);
    if (readStatus == -1) {
        onFileOpenError (MODELVALUESLOCATION);
    }
    sscanf(buffer, "%d", numberOfLayers); // convert buffer string to int and shove in numberOfLayers

    // get numberOfNeuronsTotal from file
    readStatus = getdelim(&buffer, &lineLength, VALUEDELIM, thefile);
    if (readStatus == -1) {
        onFileOpenError (MODELVALUESLOCATION);
    }
    sscanf(buffer, "%d", numberOfNeuronsTotal); // convert buffer string to int and shove in numberOfNeuronsTotal

    // get numberOfWeightsTotal from file
    readStatus = getdelim(&buffer, &lineLength, '\n', thefile);
    if (readStatus == -1) {
        onFileOpenError (MODELVALUESLOCATION);
    }
    sscanf(buffer, "%d", numberOfWeightsTotal); // convert buffer string to int and shove in numberOfNeuronsTotal

    // stretch arrays to be numberOfLayers * sizeof(type)
    int* tempPtr = (*numberOfNeuronsPerLayer); // keep track of old memory
    (*numberOfNeuronsPerLayer) = (int *) malloc((*numberOfLayers) * sizeof(int));
    if ((*numberOfNeuronsPerLayer) == NULL) {
        onMallocError((*numberOfLayers) * sizeof(int));
    }
    free(tempPtr); // release old memory

    tempPtr = (*numberOfWeightsPerLayer); // keep track of old memory
    (*numberOfWeightsPerLayer) = (int *) malloc((*numberOfLayers) * sizeof(int));
    if ((*numberOfWeightsPerLayer) == NULL) {
        onMallocError((*numberOfLayers) * sizeof(int));
    }
    free(tempPtr); // release old memory

    tempPtr = (*firstNeuronIndexPerLayer); // keep track of old memory
    (*firstNeuronIndexPerLayer) = (int *) malloc((*numberOfLayers) * sizeof(int));
    if ((*firstNeuronIndexPerLayer) == NULL) {
        onMallocError((*numberOfLayers) * sizeof(int));
    }
    free(tempPtr); // release old memory

    tempPtr = (*firstWeightIndexPerLayer); // keep track of old memory
    (*firstWeightIndexPerLayer) = (int *) malloc((*numberOfLayers) * sizeof(int));
    if ((*firstWeightIndexPerLayer) == NULL) {
        onMallocError((*numberOfLayers) * sizeof(int));
    }
    free(tempPtr); // release old memory

    // get information about each layer
    for (int i = 0; i < (*numberOfLayers); i++) {
        // get numberOfNeuronsPerLayer[i] from file
        readStatus = getdelim(&buffer, &lineLength, VALUEDELIM, thefile);
        if (readStatus == -1) {
            onFileReadError (MODELVALUESLOCATION);
        }
        sscanf(buffer, "%d", &((*numberOfNeuronsPerLayer)[i]));

        // get numberOfWeightsPerLayer[i] from file
        readStatus = getdelim(&buffer, &lineLength, VALUEDELIM, thefile);
        if (readStatus == -1) {
            onFileReadError (MODELVALUESLOCATION);
        }
        sscanf(buffer, "%d", &((*numberOfWeightsPerLayer)[i]));

        // get firstNeuronIndexPerLayer[i] from file
        readStatus = getdelim(&buffer, &lineLength, VALUEDELIM, thefile);
        if (readStatus == -1) {
            onFileReadError (MODELVALUESLOCATION);
        }
        sscanf(buffer, "%d", &((*firstNeuronIndexPerLayer)[i]));

        // get firstWeightIndexPerLayer[i] from file
        readStatus = getdelim(&buffer, &lineLength, '\n', thefile);
        if (readStatus == -1) {
            onFileReadError (MODELVALUESLOCATION);
        }
        sscanf(buffer, "%d", &((*firstWeightIndexPerLayer)[i]));
    }

    fclose(thefile); // close the file once we're done with it
} //end readModelValuesFromDisk function

/*
 * readModel
 * @params: numberOfLayers - pointer to an int value (used to store value before return)
 * @params: numberOfNeuronsTotal - pointer to an int value (used to store value before return)
 * @params: numberOfWeightsTotal - pointer to an int value (used to store value before return)
 * @params: numberOfNeuronsPerLayer - pointer to an array of int values (used to store value before return)
 * @params: numberOfWeightsPerLayer - pointer to an array of int values (used to store value before return)
 * @params: firstNeuronIndexPerLayer - pointer to an array of int values (used to store value before return)
 * @params: firstWeightIndexPerLayer - pointer to an array of int values (used to store value before return)
 * @params: weights - pointer to an array of double values (used to store value before return)
 * @params: biases - pointer to an array of double values (used to store value before return)
 * @params: learningRate - pointer to a double value (used to store value before return)
 * @params: epochs - pointer to an int value (used to store value before return)
 */
void readModel(int* numberOfLayers, int* numberOfNeuronsTotal, int* numberOfWeightsTotal, int** numberOfNeuronsPerLayer, int** numberOfWeightsPerLayer,
        int** firstNeuronIndexPerLayer, int** firstWeightIndexPerLayer, double** weights, double** biases, double* learningRate, int* epochs) {
    // verify directory containing model exists
    if (stat(MODELDIRECTORY, &st) == -1) {
        onFileOpenError (MODELDIRECTORY);
    }
    readModelValuesFromDisk(numberOfLayers, numberOfNeuronsTotal, numberOfWeightsTotal, numberOfNeuronsPerLayer, numberOfWeightsPerLayer,
            firstNeuronIndexPerLayer, firstWeightIndexPerLayer);
    readWeightsFromDisk(weights, *numberOfWeightsTotal);
    readBiasesFromDisk(biases, *numberOfNeuronsTotal); // num biases total is equal to num neurons total
    readEpochsFromDisk(epochs);
    readLearningRateFromDisk(learningRate);
} //end readModel function
