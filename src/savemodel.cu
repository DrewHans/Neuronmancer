/*******************************************************************************************
 * Filename: savemodel.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the functions for saving a model to disk.
 *******************************************************************************************
 */

#define VALUEDELIM ","
#define MODELVALUESLOCATION "./nmModel/modelvalues.csv"
#define WEIGHTSFILELOCATION "./nmModel/weights.csv"
#define BIASESFILELOCATION "./nmModel/biases.csv"
#define EPOCHSFILELOCATION "./nmModel/epochs.txt"
#define LEARNINGRATEFILELOCATION "./nmModel/learningrate.txt"

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

struct stat st = {0}; // needed for using stat



void saveBiasesToDisk(double* biases, int numberOfBiasesTotal) {
    FILE* thefile = fopen(BIASESFILELOCATION, "w");
    if (thefile == NULL) {
        onFileOpenError(BIASESFILELOCATION);
    }

    for (int i = 0; i < numberOfBiasesTotal; i++) {
        fprintf(thefile, "%lf", biases[i]);   // write long float (double) to file
        fprintf(thefile, "%s", VALUEDELIM);   // write delimiter to file
    }

    fclose(thefile); // close the file once we're done with it
}//end saveBiasesToDisk function

void saveEpochsToDisk(int epochs) {
    FILE* thefile = fopen(EPOCHSFILELOCATION, "w");
    if (thefile == NULL) {
        onFileOpenError(EPOCHSFILELOCATION);
    }

    fprintf(thefile, "%d", epochs); // write int to file
    fclose(thefile); // close the file once we're done with it
}//end saveEpochsToDisk function

void saveLearningRateToDisk(double learningRate) {
    FILE* thefile = fopen(LEARNINGRATEFILELOCATION, "w");
    if (thefile == NULL) {
        onFileOpenError(LEARNINGRATEFILELOCATION);
    }

    fprintf(thefile, "%lf", learningRate); // write long float (double) to file
    fclose(thefile); // close the file once we're done with it
}//end saveLearningRateToDisk function

void saveWeightsToDisk(double* weights, int numberOfWeightsTotal) {
    FILE* thefile = fopen(WEIGHTSFILELOCATION, "w");
    if (thefile == NULL) {
        onFileOpenError(WEIGHTSFILELOCATION);
    }

    for (int i = 0; i < numberOfWeightsTotal; i++) {
        fprintf(thefile, "%lf", weights[i]);  // write long float (double) to file
        fprintf(thefile, "%s", VALUEDELIM);   // write delimiter to file
    }

    fclose(thefile); // close the file once we're done with it
}//end saveWeightsToDisk function

void saveModelValuesToDisk(int numberOfLayers, int numberOfNeuronsTotal, int numberOfWeightsTotal, \
                           int* numberOfNeuronsPerLayer, int* numberOfWeightsPerLayer, \
                           int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer) {
    FILE* thefile = fopen(MODELVALUESLOCATION, "w");
    if (thefile == NULL) {
        onFileOpenError(MODELVALUESLOCATION);
    }

    fprintf(thefile, "%d", numberOfLayers);       // write int to file
    fprintf(thefile, "%s", VALUEDELIM);           // write delimiter to file
    fprintf(thefile, "%d", numberOfNeuronsTotal); // write int to file
    fprintf(thefile, "%s", VALUEDELIM);           // write delimiter to file
    fprintf(thefile, "%d", numberOfWeightsTotal); // write int to file
    fprintf(thefile, "\n");                       // write newline to file

    for (int i = 0; i < numberOfLayers; i++) {
        fprintf(thefile, "%d", numberOfNeuronsPerLayer[i]);  // write int to file
        fprintf(thefile, "%s", VALUEDELIM);                  // write delimiter to file
        fprintf(thefile, "%d", numberOfWeightsPerLayer[i]);  // write int to file
        fprintf(thefile, "%s", VALUEDELIM);                  // write delimiter to file
        fprintf(thefile, "%d", firstNeuronIndexPerLayer[i]); // write int to file
        fprintf(thefile, "%s", VALUEDELIM);                  // write delimiter to file
        fprintf(thefile, "%d", firstWeightIndexPerLayer[i]); // write int to file
        fprintf(thefile, "\n");                              // write newline to file (indicates new layer)
    }

    fclose(thefile); // close the file once we're done with it
}//end saveModelValuesToDisk function

void saveModel(int numberOfLayers, int numberOfNeuronsTotal, int numberOfWeightsTotal, \
               int* numberOfNeuronsPerLayer, int* numberOfWeightsPerLayer, \
               int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer, \
               double* weights, double* biases, double learningRate, int epochs) {
    // make directory to store the model
    if (stat("./nmModel", &st) == -1) {
        mkdir("./nmModel", 0700);
    }
    saveModelValuesToDisk(numberOfLayers, numberOfNeuronsTotal, numberOfWeightsTotal, \
                          numberOfNeuronsPerLayer, numberOfWeightsPerLayer, \
                          firstNeuronIndexPerLayer, firstWeightIndexPerLayer);
    saveWeightsToDisk(weights, numberOfWeightsTotal);
    saveBiasesToDisk(biases, numberOfBiasesTotal);
    saveEpochsToDisk(epochs);
    saveLearningRateToDisk(learningRate);
}//end saveModel function
