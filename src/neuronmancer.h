/*******************************************************************************************
 * Filename: neuronmancer.h
 * Author: Drew Hans (github.com/drewhans555)
 * Description: Header file for main.cu
 *******************************************************************************************
 */

#ifndef NEURONMANCER_H
#define NEURONMANCER_H

// define delimiters and locations for readmodel.cu and savemodel.cu
#define VALUEDELIM ','
#define VALUEDELIMCHARSTAR ","
#define MODELDIRECTORY "./nmModel"
#define MODELVALUESLOCATION "./nmModel/modelvalues.csv"
#define WEIGHTSFILELOCATION "./nmModel/weights.csv"
#define BIASESFILELOCATION "./nmModel/biases.csv"
#define EPOCHSFILELOCATION "./nmModel/epochs.txt"
#define LEARNINGRATEFILELOCATION "./nmModel/learningrate.txt"

// define single sample size, set sizes, and locations for mnist dataset
#define MNISTSAMPLEDATASIZE 784
#define MNISTTESTSETSIZE 10000
#define MNISTTRAININGSETSIZE 60000
#define MNISTTESTFILELOCATION "../mnist/mnist_test.csv"
#define MNISTTRAINFILELOCATION "../mnist/mnist_train.csv"

// define enum for available activation functions
typedef enum {
    SIGMACT, RELUACT, TANHACT
} Activation;

// define struct for using the stat command
struct stat st = { 0 };

// define function prototypes for activationfunctions.cu
__host__ __device__ double sigmoidFunction(double d);
__host__ __device__ double sigmoidDerivative(double d);
__host__ __device__ double tanhFunction(double d);
__host__ __device__ double tanhDerivative(double d);
__host__ __device__ double reluFunction(double d);
__host__ __device__ double reluDerivative(double d);
__global__ void sigmoidKernel(double* devNeurons, int neuronIndexStart, int numberOfNeuronsInLayer);
__global__ void reluKernel(double* devNeurons, int neuronIndexStart, int numberOfNeuronsInLayer);
__global__ void tanhKernel(double* devNeurons, int neuronIndexStart, int numberOfNeuronsInLayer);

// define function prototypes for backpropagationfunctions.cu
__global__ void backpropagateErrorsKernel(double* devNeurons, double* devWeights, double* devBiases, double* devNeuronErrors, int numberOfNeuronsInLeftLayer,
        int numberOfWeightsBetweenLayers, int indexOfFirstNeuronInLeft, int indexOfFirstNeuronInRight, int indexOfFirstWeight);
void backpropagateWithDevice(int numBlocks, int threadsPerBlock, double* devExpectedOutput, double* devNeurons, double* devWeights, double* devBiases,
        double* devNeuronErrors, int numberOfLayers, int* neuronsPerLayer, int* weightsPerLayer, int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer);
void backpropagateWithHost(double* expectedOutput, double* neurons, double* weights, double* biases, double* neuronErrors, int numberOfLayers,
        int* neuronsPerLayer, int* weightsPerLayer, int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer);

// define function prototypes for combinationfunctions.cu
__global__ void combinationFunctionKernel(double* devNeurons, double* devWeights, double* devBiases, int neuronIndexStart, int prevLayerNeuronIndexStart,
        int weightIndexStart, int numberOfNeuronsInLayer, int numberOfNeuronsInPrevLayer);
void combinationFunction(double* neurons, double* weights, double* biases, int neuronIndex, int prevLayerIndexStart, int weightIndexStart, int prevLayerSize);

// define function prototypes for costfunctions.cu
__global__ void costFunctionKernel(double* devExpectedOutput, double* devNeurons, double* devNeuronErrors, int neuronIndexStart, int numberOfNeuronsInLayer);
double costFunction(double* expectedValue, double* calculatedValue);

// define function prototypes for feedforwardfunctions.cu
void feedforwardWithDevice(int numBlocks, int threadsPerBlock, double* devNeurons, double* devWeights, double* devBiases, int numberOfLayers,
        int* numberOfNeuronsPerLayer, int* numberOfWeightsPerLayer, int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer);
void feedforwardWithHost(double* neurons, double* weights, double* biases, int numberOfLayers, int* neuronsPerLayer, int* weightsPerLayer,
        int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer);

// define function prototypes for helperfunctions.cu
void getDeviceProperties(int* multiProcessorCount, int* warpSize);
void initArrayToRandomDoubles(double* a, int n);
void initArrayToZeros(double* a, int n);
void loadRandomInput(double* neurons, int n);
void printarray(const char* name, double* array, int n);
void printFarewellMSG();
void onFileOpenError(const char* path);
void onInvalidInput(int myPatience);
void onMallocError(int size);

// define function prototypes for loadmnist.cu
void loadMnistTestSamples(unsigned char* testData, char* testLabels, int* numberOfSamples);
void loadMnistTrainingSamples(unsigned char* trainingData, char* trainingLabels, int* numberOfSamples);
void loadNextMnistSampleData(double* neurons, const unsigned char* mnistData, int mnistSampleDataIndexStart);
__global__ void loadNextMnistSampleDataKernel(double* devNeurons, const unsigned char* devMnistData, int mnistSampleDataIndexStart);
void loadNextMnistSampleLabel(double* outputExpected, const char* mnistLabels, int mnistSampleLabelIndex);

// define function prototypes for readmodel.cu
void readBiasesFromDisk(double* biases, int numberOfBiasesTotal);
void readEpochsFromDisk(int* epochs);
void readLearningRateFromDisk(double* learningRate);
void readWeightsFromDisk(double* weights, int numberOfWeightsTotal);
void readModelValuesFromDisk(int* numberOfLayers, int* numberOfNeuronsTotal, int* numberOfWeightsTotal, int* numberOfNeuronsPerLayer,
        int* numberOfWeightsPerLayer, int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer);
void readModel(int* numberOfLayers, int* numberOfNeuronsTotal, int* numberOfWeightsTotal, int* numberOfNeuronsPerLayer, int* numberOfWeightsPerLayer,
        int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer, double* weights, double* biases, double* learningRate, int* epochs);

// define function prototypes for savemodel.cu
void saveBiasesToDisk(double* biases, int numberOfBiasesTotal);
void saveEpochsToDisk(int epochs);
void saveLearningRateToDisk(double learningRate);
void saveWeightsToDisk(double* weights, int numberOfWeightsTotal);
void saveModelValuesToDisk(int numberOfLayers, int numberOfNeuronsTotal, int numberOfWeightsTotal, int* numberOfNeuronsPerLayer, int* numberOfWeightsPerLayer,
        int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer);
void saveModel(int numberOfLayers, int numberOfNeuronsTotal, int numberOfWeightsTotal, int* numberOfNeuronsPerLayer, int* numberOfWeightsPerLayer,
        int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer, double* weights, double* biases, double learningRate, int epochs);

// define function prototypes for ui_create.cu
void ui_create();

// define function prototypes for ui_evaluate.cu
void ui_evaluate();

// define function prototypes for ui_train.cu
void ui_train();

// define function prototypes for updatebiasesfunctions.cu
__global__ void biasUpdateKernel(double* devNeurons, double* devBiases, double* devNeuronErrors, int numberOfNeuronsTotal, double learningRate);
void updateBiases(double* neurons, double* biases, double* neuronErrors, int numberOfNeuronsTotal, double learningRate);

// define function prototypes for updateweightsfunctions.cu
__global__ void weightUpdateKernel(double* devNeurons, double* devWeights, double* devNeuronErrors, int numberOfNeuronsInLeftLayer,
        int numberOfNeuronsInRightLayer, int numberOfWeightsBetweenLayers, int indexOfFirstNeuronInLeft, int indexOfFirstWeight, double learningRate);
void updateWeights(double* neurons, double* weights, double* neuronErrors, int numberOfLayers, int* neuronsPerLayer, int* firstNeuronIndexPerLayer,
        int* firstWeightIndexPerLayer, double learningRate);
#endif
