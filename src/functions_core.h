/*******************************************************************************
 * Filename: functions_core.h
 * Author: Drew Hans (github.com/drewhans555)
 * Description: functions_core.c's header file - contains function prototypes.
 *******************************************************************************
 */

#ifndef FUNCTIONS_CORE_H
#define FUNCTIONS_CORE_H

// define function prototypes for functions_core.c
void calculateHiddenLayerDeltas(HiddenLayer *hl, OutputLayer *ol);

void calculateOutputLayerDeltas(OutputLayer *ol, ExpectedOutput *expected);

void feedInputLayer(InputLayer *il, MNIST_Image *image);

void feedHiddenLayer(HiddenLayer *hl, InputLayer *il);

void feedHLNeuron(HLNeuron *hln, InputLayer *il);

int feedNetwork(
    InputLayer *il,
    HiddenLayer *hl,
    OutputLayer *ol,
    MNIST_Image *image);

void feedOLNeuron(OLNeuron *oln, HiddenLayer *hl);

void feedOutputLayer(OutputLayer *ol, HiddenLayer *hl);

ExpectedOutput getExpectedOutput(int mnistLabel);

int getNetworkPrediction(OutputLayer *ol);

void initNetwork(HiddenLayer *hl, OutputLayer *ol);

float sigmoid(const float x);

float sigmoidPrime(const float x);

void train(InputLayer *il, HiddenLayer *hl, OutputLayer *ol);

void trainNetwork(
    InputLayer *il,
    HiddenLayer *hl,
    OutputLayer *ol,
    MNIST_Image *image,
    int target);

void updateHLNeuronWeightsAndBiases(HLNeuron *hln, InputLayer *il);

void updateNetworkWeightsAndBiases(
    InputLayer *il,
    HiddenLayer *hl,
    OutputLayer *ol);

void updateOLNeuronWeightsAndBiases(OLNeuron *oln, HiddenLayer *hl);

#endif /* FUNCTIONS_CORE_H */
