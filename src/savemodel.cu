/*******************************************************************************************
 * Filename: savemodel.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the functions for saving a model to disk.
 *******************************************************************************************
 */

/*
 * saveBiasesToDisk
 * @params: biases - pointer to an array of double values
 * @params: numberOfBiasesTotal - equal to numberOfNeuronsTotal
 */
void saveBiasesToDisk(double* biases, int numberOfBiasesTotal) {
    FILE* thefile = fopen(BIASESFILELOCATION, "w");
    if (thefile == NULL) {
        onFileOpenError (BIASESFILELOCATION);
    }

    for (int i = 0; i < numberOfBiasesTotal; i++) {
        fprintf(thefile, "%lf", biases[i]);   // write long float (double) to file
        fprintf(thefile, "%s", VALUEDELIMCHARSTAR);   // write delimiter to file
    }

    fprintf(thefile, "\n"); // write newline to file
    fclose(thefile); // close the file once we're done with it
} //end saveBiasesToDisk function

/*
 * saveEpochsToDisk
 * @params: epochs - the number of training cycles (in one cycle every sample in the data set is fed in and then errors "backpropagate")
 */
void saveEpochsToDisk(int epochs) {
    FILE* thefile = fopen(EPOCHSFILELOCATION, "w");
    if (thefile == NULL) {
        onFileOpenError (EPOCHSFILELOCATION);
    }

    fprintf(thefile, "%d", epochs); // write int to file
    fprintf(thefile, "\n"); // write newline to file
    fclose(thefile); // close the file once we're done with it
} //end saveEpochsToDisk function

/*
 * saveLearningRateToDisk
 * @params: learningRate - the rate at which we want our network to make adjustments to the weights
 */
void saveLearningRateToDisk(double learningRate) {
    FILE* thefile = fopen(LEARNINGRATEFILELOCATION, "w");
    if (thefile == NULL) {
        onFileOpenError (LEARNINGRATEFILELOCATION);
    }

    fprintf(thefile, "%lf", learningRate); // write long float (double) to file
    fprintf(thefile, "\n"); // write newline to file
    fclose(thefile); // close the file once we're done with it
} //end saveLearningRateToDisk function

/*
 * saveWeightsToDisk
 * @params: weights - pointer to an array of double values
 * @params: numberOfWeightsTotal - the total number of weights in the model
 */
void saveWeightsToDisk(double* weights, int numberOfWeightsTotal) {
    FILE* thefile = fopen(WEIGHTSFILELOCATION, "w");
    if (thefile == NULL) {
        onFileOpenError (WEIGHTSFILELOCATION);
    }

    for (int i = 0; i < numberOfWeightsTotal; i++) {
        fprintf(thefile, "%lf", weights[i]);  // write long float (double) to file
        fprintf(thefile, "%s", VALUEDELIMCHARSTAR);   // write delimiter to file
    }

    fprintf(thefile, "\n"); // write newline to file
    fclose(thefile); // close the file once we're done with it
} //end saveWeightsToDisk function

/*
 * saveModelValuesToDisk
 * @params: numberOfLayers - the number of layers in the model
 * @params: numberOfNeuronsTotal - the total number of neurons in the model
 * @params: numberOfWeightsTotal - the total number of weights in the model
 * @params: numberOfNeuronsPerLayer - pointer to an array of int values (the number of neurons in each layer in the model)
 * @params: numberOfWeightsPerLayer - pointer to an array of int values (the number of weights in each layer in the model)
 * @params: firstNeuronIndexPerLayer - pointer to an array of int values (the indexes of each layer's first neuron)
 * @params: firstWeightIndexPerLayer - pointer to an array of int values (the indexes of each layer's first weight)
 */
void saveModelValuesToDisk(int numberOfLayers, int numberOfNeuronsTotal, int numberOfWeightsTotal, int* numberOfNeuronsPerLayer, int* numberOfWeightsPerLayer,
        int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer) {
    FILE* thefile = fopen(MODELVALUESLOCATION, "w");
    if (thefile == NULL) {
        onFileOpenError (MODELVALUESLOCATION);
    }

    fprintf(thefile, "%d", numberOfLayers);       // write int to file
    fprintf(thefile, "%s", VALUEDELIMCHARSTAR);           // write delimiter to file
    fprintf(thefile, "%d", numberOfNeuronsTotal); // write int to file
    fprintf(thefile, "%s", VALUEDELIMCHARSTAR);           // write delimiter to file
    fprintf(thefile, "%d", numberOfWeightsTotal); // write int to file
    fprintf(thefile, "\n");                       // write newline to file

    for (int i = 0; i < numberOfLayers; i++) {
        fprintf(thefile, "%d", numberOfNeuronsPerLayer[i]);  // write int to file
        fprintf(thefile, "%s", VALUEDELIMCHARSTAR);                  // write delimiter to file
        fprintf(thefile, "%d", numberOfWeightsPerLayer[i]);  // write int to file
        fprintf(thefile, "%s", VALUEDELIMCHARSTAR);                  // write delimiter to file
        fprintf(thefile, "%d", firstNeuronIndexPerLayer[i]); // write int to file
        fprintf(thefile, "%s", VALUEDELIMCHARSTAR);                  // write delimiter to file
        fprintf(thefile, "%d", firstWeightIndexPerLayer[i]); // write int to file
        fprintf(thefile, "\n");                              // write newline to file (indicates new layer)
    }

    fclose(thefile); // close the file once we're done with it
} //end saveModelValuesToDisk function

/*
 * saveModel
 * @params: numberOfLayers - the number of layers in the model
 * @params: numberOfNeuronsTotal - the total number of neurons in the model
 * @params: numberOfWeightsTotal - the total number of weights in the model
 * @params: numberOfNeuronsPerLayer - pointer to an array of int values (the number of neurons in each layer in the model)
 * @params: numberOfWeightsPerLayer - pointer to an array of int values (the number of weights in each layer in the model)
 * @params: firstNeuronIndexPerLayer - pointer to an array of int values (the indexes of each layer's first neuron)
 * @params: firstWeightIndexPerLayer - pointer to an array of int values (the indexes of each layer's first weight)
 * @params: weights - pointer to an array of double values
 * @params: biases - pointer to an array of double values
 * @params: learningRate - the rate at which we want our network to make adjustments to the weights
 * @params: epochs - the number of training cycles (in one cycle every sample in the data set is fed in and then errors "backpropagate")
 */
void saveModel(int numberOfLayers, int numberOfNeuronsTotal, int numberOfWeightsTotal, int* numberOfNeuronsPerLayer, int* numberOfWeightsPerLayer,
        int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer, double* weights, double* biases, double learningRate, int epochs) {
    // make directory to store the model
    if (stat(MODELDIRECTORY, &st) == -1) {
        mkdir(MODELDIRECTORY, 0700);
    }
    saveModelValuesToDisk(numberOfLayers, numberOfNeuronsTotal, numberOfWeightsTotal, numberOfNeuronsPerLayer, numberOfWeightsPerLayer,
            firstNeuronIndexPerLayer, firstWeightIndexPerLayer);
    saveWeightsToDisk(weights, numberOfWeightsTotal);
    saveBiasesToDisk(biases, numberOfNeuronsTotal); // num biases total is equal to num neurons total
    saveEpochsToDisk(epochs);
    saveLearningRateToDisk(learningRate);
} //end saveModel function
