/*******************************************************************************************
 * Filename: model_read.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the functions for reading a model from disk.
 *******************************************************************************************
 */

/*
 * readBiasesFromDisk - read in model's bias values from a file on disk
 * @params: numberOfBiasesTotal - the int number of biases total (equal to numberOfNeuronsTotal)
 * @return: a float pointer-pointer to the array of bias values read from disk
 */
float** readBiasesFromDisk(unsigned int numberOfBiasesTotal) {
    FILE* thefile = fopen(BIASESFILELOCATION, "r");
    if (thefile == NULL) {
        onFileOpenError(BIASESFILELOCATION);
    }

    // initialize an array to hold the biases
    float* p_biases = (float*) malloc(numberOfBiasesTotal * sizeof(float));
    if (p_biases == NULL) {
        onMallocError(numberOfBiasesTotal * sizeof(float));
    }

    // setup variables needed for getline function
    size_t characters;
    size_t bufsize = numberOfBiasesTotal * 8;
    char* buffer = (char*) malloc(bufsize * sizeof(char));
    if (buffer == NULL) {
        onMallocError(bufsize * sizeof(char));
    }

    // start at 0, increment after storing each bias
    int biasIndex = 0;

    // read line from thefile, stop after reading all lines
    while ((characters = getline(&buffer, &bufsize, thefile)) != -1) {
        char* token = strtok(buffer, VALUEDELIM); // grab first token from the line
        while (token) {
            sscanf(token, "%f", &(p_biases[biasIndex])); // store token
            token = strtok(NULL, VALUEDELIM); // grab next token from line
            biasIndex++; // increment index after storing each bias
        }
    }

    fclose(thefile); // close the file once we're done with it

    return &p_biases;
} //end readBiasesFromDisk function

/*
 * readEpochsFromDisk - read in model's epochs value from a file on disk
 * @return: the int number of epochs
 */
unsigned int readEpochsFromDisk() {
    FILE* thefile = fopen(EPOCHSFILELOCATION, "r");
    if (thefile == NULL) {
        onFileOpenError(EPOCHSFILELOCATION);
    }

    // initialize variable to hold epochs
    unsigned int epochs = 0;

    // setup variables needed for getline function
    size_t characters;
    size_t bufsize = 16;
    char* buffer = (char*) malloc(bufsize * sizeof(char));
    if (buffer == NULL) {
        onMallocError(bufsize * sizeof(char));
    }

    // read line from thefile and store 
    characters = getline(&buffer, &bufsize, thefile);
    sscanf(buffer, "%u", &(epochs));

    fclose(thefile); // close the file once we're done with it

    return epochs;
} //end readEpochsFromDisk function

/*
 * readLearningRateFromDisk - read in model's learning rate value from a file on disk
 * @return: the float learning rate
 */
float readLearningRateFromDisk() {
    FILE* thefile = fopen(LEARNINGRATEFILELOCATION, "r");
    if (thefile == NULL) {
        onFileOpenError(LEARNINGRATEFILELOCATION);
    }

    // initialize variable to hold learning rate
    float learningRate = 0.0;

    // setup variables needed for getline function
    size_t characters;
    size_t bufsize = 16;
    char* buffer = (char*) malloc(bufsize * sizeof(char));
    if (buffer == NULL) {
        onMallocError(bufsize * sizeof(char));
    }

    // read line from thefile and store 
    characters = getline(&buffer, &bufsize, thefile);
    sscanf(buffer, "%f", &(learningRate));

    fclose(thefile); // close the file once we're done with it

    return learningRate;
} //end readLearningRateFromDisk function

/*
 * readWeightsFromDisk - read in model's weight values from a file on disk
 * @params: numberOfWeightsTotal - the int number of weights total
 * @return: a float pointer-pointer to the array of weight values read from disk
 */
float** readWeightsFromDisk(unsigned int numberOfWeightsTotal) {
    FILE* thefile = fopen(WEIGHTSFILELOCATION, "r");
    if (thefile == NULL) {
        onFileOpenError(WEIGHTSFILELOCATION);
    }

    // initialize an array to hold the biases
    float* p_weights = (float*) malloc(numberOfWeightsTotal * sizeof(float));
    if (p_weights == NULL) {
        onMallocError(numberOfBiasesTotal * sizeof(float));
    }

    // setup variables needed for getline function
    size_t characters;
    size_t bufsize = numberOfWeightsTotal * 8;
    char* buffer = (char*) malloc(bufsize * sizeof(char));
    if (buffer == NULL) {
        onMallocError(bufsize * sizeof(char));
    }

    // start at 0, increment after storing each bias
    unsigned int weightIndex = 0;

    // read line from thefile, stop after reading all lines
    while ((characters = getline(&buffer, &bufsize, thefile)) != -1) {
        char* token = strtok(buffer, VALUEDELIM); // grab first token from the line
        while (token) {
            sscanf(token, "%f", &(p_weights[weightIndex])); // store token
            token = strtok(NULL, VALUEDELIM); // grab next token from line
            weightIndex++; // increment index after storing each bias
        }
    }

    fclose(thefile); // close the file once we're done with it

    return &p_weights;
} //end readWeightsFromDisk function

/*
 * readModelValuesFromDisk - read in model's structure values from a file on disk
 * @params: p_numberOfLayers - the int pointer to the variable holding the number of layers in the model
 * @params: p_numberOfNeuronsTotal - the int pointer to the variable holding the number of neurons in the model
 * @params: p_numberOfWeightsTotal - the int pointer to the variable holding the number of weights in the model
 * @params: p_numberOfNeuronsInLayer - the int pointer-pointer to the array holding the number of neurons in each layer in the model
 * @params: p_numberOfWeightsInFrontOfLayer - the int pointer-pointer to the array holding the number of weights in front of each layer in the model
 * @params: p_indexOfFirstNeuronInLayer - the int pointer-pointer to the array holding the  indexes of the first neuron in each layer
 * @params: p_indexOfFirstWeightInFrontOfLayer - the int pointer-pointer to the array holding the  indexes of the first weight in front of each layer
 */
void readModelValuesFromDisk(unsigned int* p_numberOfLayers, unsigned int* p_numberOfNeuronsTotal, unsigned int* p_numberOfWeightsTotal, 
                             unsigned int** p_numberOfNeuronsInLayer, unsigned int** p_numberOfWeightsInFrontOfLayer, 
                             unsigned int** p_indexOfFirstNeuronInLayer, unsigned int** p_indexOfFirstWeightInFrontOfLayer) {
    FILE* thefile = fopen(MODELVALUESLOCATION, "r");
    if (thefile == NULL) {
        onFileOpenError(MODELVALUESLOCATION);
    }

    // setup variables needed for getline function
    size_t characters;
    size_t bufsize = 100;
    char* buffer = (char*) malloc(bufsize * sizeof(char));
    if (buffer == NULL) {
        onMallocError(bufsize * sizeof(char));
    }

    // read line from thefile and store the model-structure values
    characters = getline(&buffer, &bufsize, thefile);
    char* token = strtok(buffer, VALUEDELIM); // grab first token from the line
    sscanf(token, "%u", p_numberOfLayers); // store token (should be numberOfLayers)
    token = strtok(NULL, VALUEDELIM); // grab next token from line
    sscanf(token, "%u", p_numberOfNeuronsTotal); // store token (should be numberOfNeuronsTotal)
    token = strtok(NULL, VALUEDELIM); // grab next token from line
    sscanf(token, "%u", p_numberOfWeightsTotal); // store token (should be numberOfWeightsTotal)

    // stretch p_numberOfNeuronsInLayer array to be numberOfLayers * sizeof(type)
    unsigned int* tempPtr = (*p_numberOfNeuronsInLayer); // keep track of old memory
    (*p_numberOfNeuronsInLayer) = (unsigned int *) malloc((*numberOfLayers) * sizeof(int));
    if ((*p_numberOfNeuronsInLayer) == NULL) {
        onMallocError((*numberOfLayers) * sizeof(int));
    }
    free(tempPtr); // release old memory

    // stretch p_numberOfWeightsInFrontOfLayer array to be numberOfLayers * sizeof(type)
    tempPtr = (*p_numberOfWeightsInFrontOfLayer); // keep track of old memory
    (*p_numberOfWeightsInFrontOfLayer) = (unsigned int *) malloc((*numberOfLayers) * sizeof(int));
    if ((*p_numberOfWeightsInFrontOfLayer) == NULL) {
        onMallocError((*numberOfLayers) * sizeof(int));
    }
    free(tempPtr); // release old memory

    // stretch p_indexOfFirstNeuronInLayer array to be numberOfLayers * sizeof(type)
    tempPtr = (*p_indexOfFirstNeuronInLayer); // keep track of old memory
    (*p_indexOfFirstNeuronInLayer) = (unsigned int *) malloc((*numberOfLayers) * sizeof(int));
    if ((*p_indexOfFirstNeuronInLayer) == NULL) {
        onMallocError((*numberOfLayers) * sizeof(int));
    }
    free(tempPtr); // release old memory

    // stretch p_indexOfFirstWeightInFrontOfLayer array to be numberOfLayers * sizeof(type)
    tempPtr = (*p_indexOfFirstWeightInFrontOfLayer); // keep track of old memory
    (*p_indexOfFirstWeightInFrontOfLayer) = (unsigned int *) malloc((*numberOfLayers) * sizeof(int));
    if ((*p_indexOfFirstWeightInFrontOfLayer) == NULL) {
        onMallocError((*numberOfLayers) * sizeof(int));
    }
    free(tempPtr); // release old memory

    // start at 0, increment after storing all layer values
    unsigned int layerIndex = 0;

    // read line from thefile, stop after reading all lines
    while ((characters = getline(&buffer, &bufsize, thefile)) != -1) {
        token = strtok(buffer, VALUEDELIM); // grab first token from the line
        while (token) {
            sscanf(token, "%u", &((*numberOfNeuronsInLayer)[layerIndex])); // store token
            token = strtok(NULL, VALUEDELIM); // grab next token from line
            sscanf(token, "%u", &((*numberOfWeightsInFrontOfLayer)[layerIndex])); // store token
            token = strtok(NULL, VALUEDELIM); // grab next token from line
            sscanf(token, "%u", &((*indexOfFirstNeuronInLayer)[layerIndex])); // store token
            token = strtok(NULL, VALUEDELIM); // grab next token from line
            sscanf(token, "%u", &((*indexOfFirstWeightInFrontOfLayer)[layerIndex])); // store token
            token = strtok(NULL, VALUEDELIM); // grab next token from line
        }
    }
} //end readModelValuesFromDisk function

/*
 * readModel
 * @params: p_learningRate - a float pointer, should point to a variable holding the learning rate
 * @params: p_epochs - an int pointer, should point to a variable holding the epochs value
 * @params: p_numberOfLayers - an int pointer, should point to the variable holding the number of layers
 * @params: p_numberOfNeuronsTotal - an int pointer, should point to the variable holding the number of neurons
 * @params: p_numberOfWeightsTotal - an int pointer, should point to the variable holding the number of weights
 * @params: p_numberOfNeuronsInLayer - an int pointer-pointer, will point to the array holding the number of 
 *                                     neurons in each layer before return
 * @params: p_numberOfWeightsInFrontOfLayer - an int pointer-pointer, will point to the array holding the number 
 *                                            of weights in front of each layer before return
 * @params: p_indexOfFirstNeuronInLayer - an int pointer-pointer, will point to the array holding the indexes of 
 *                                        the first neuron in each layer before return
 * @params: p_indexOfFirstWeightInFrontOfLayer - an int pointer-pointer, will point to the array holding the indexes
 *                                               of the first weight in front of each layer before return
 * @params: p_weights - a float pointer-pointer, will point to the array of weight values before return
 * @params: p_biases - a float pointer-pointer, will point to the array of bias values before return
 */
void readModel(float* p_learningRate, unsigned int* p_epochs, 
               unsigned int* p_numberOfLayers, unsigned int* p_numberOfNeuronsTotal, unsigned int* p_numberOfWeightsTotal, 
               unsigned int** p_numberOfNeuronsInLayer, unsigned int** p_numberOfWeightsInFrontOfLayer,
               unsigned int** p_indexOfFirstNeuronInLayer, unsigned int** p_indexOfFirstWeightInFrontOfLayer, 
               float** p_weights, float** p_biases) {
    // verify directory containing model exists
    if (stat(MODELDIRECTORY, &st) == -1) {
        onFileOpenError(MODELDIRECTORY);
    }

    // read model's learning rate from a file on disk
    (*p_learningRate) = readLearningRateFromDisk();

    // read model's epochs from a file on disk
    (*p_epochs) = readEpochsFromDisk();

    // read model's structure values from a file on disk
    readModelValuesFromDisk(p_numberOfLayers, p_numberOfNeuronsTotal, p_numberOfWeightsTotal, 
                            p_numberOfNeuronsInLayer, p_numberOfWeightsInFrontOfLayer,
                            p_indexOfFirstNeuronInLayer, p_indexOfFirstWeightInFrontOfLayer);

    // read model's weight values from a file on disk
    float* tempFloatPtr = (*p_weights); // keep track of old memory
    p_weights = readWeightsFromDisk((*p_numberOfWeightsTotal));
    free(tempFloatPtr); // release old memory

    // read model's bias values from a file on disk
    tempFloatPtr = (*p_biases); // keep track of old memory
    p_biases = readWeightsFromDisk((*p_numberOfNeuronsTotal));
    free(tempFloatPtr); // release old memory

} //end readModel function
