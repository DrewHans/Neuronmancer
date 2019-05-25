/********************************************************************************
 * Filename: functions_core.c
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the functions needed to train and evaluate a
 *              network on the host machine (using the CPU).
 ********************************************************************************
 */

#include "main.h"

/*
 * calculateHiddenLayerDeltas - calculates the delta values for hidden-layer neurons
 * @params HiddenLayer* hl - pointer to a HiddenLayer struct
 * @params OutputLayer* ol - pointer to an OutputLayer struct
 */
void calculateHiddenLayerDeltas(HiddenLayer* hl, OutputLayer* ol) {

    // for each hNeuron in HiddenLayer
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        hl->hNeuron[i].delta = 0.0; // clear out any garbage

        // for each oNeuron in OutputLayer
        for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
            // propagate ol->oNeuron[j]'s delta backwards
            hl->hNeuron[i].delta += ol->oNeuron[j].weight[i] * ol->oNeuron[j].delta;
        }

        // calculate hl->hNeuron[i]'s delta
        hl->hNeuron[i].delta = sigmoidPrime(hl->hNeuron[i].weightedSum) * hl->hNeuron[i].delta;
    }

} //end calculateHiddenLayerDeltas function

/*
 * calculateOutputLayerDeltas - calculates the delta values for output-layer neurons
 * @params OutputLayer* ol - pointer to an OutputLayer struct
 * @params ExpectedOutput* expected - pointer to an ExpectedOutput struct
 */
void calculateOutputLayerDeltas(OutputLayer* ol, ExpectedOutput* expected) {

    // for each oNeuron in OutputLayer
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        ol->oNeuron[i].delta = sigmoidPrime(ol->oNeuron[i].weightedSum) * (expected->value[i] - ol->oNeuron[i].output);
    }

} //end calculateOutputLayerDeltas function

/*
 * feedInputLayer - feeds MNIST pixel values into the input-layer
 * @params InputLayer* il - pointer to an InputLayer struct
 * @params MNIST_Image* image - pointer to an MNIST_Image struct
 */
void feedInputLayer(InputLayer* il, MNIST_Image* image) {

    // for each il->input[i]
    for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
        // if image->pixel[i] !0 then set il->input[i] to 1, else set to 0
        il->input[i] = (image->pixel[i] ? 1 : 0);
    }

} //end feedInputLayer function

/*
 * feedHiddenLayer - feeds input-layer values into hidden-layer
 * @params HiddenLayer* hl - pointer to a HiddenLayer struct
 * @params InputLayer* il - pointer to an InputLayer struct
 */
void feedHiddenLayer(HiddenLayer* hl, InputLayer* il) {

    // for each hl->hNeuron[i], feedHLNeuron
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        feedHLNeuron(&(hl->hNeuron[i]), il);
    }

} //end feedHiddenLayer function

/*
 * feedHLNeuron - feeds input-layer values into a hidden-layer neuron
 * @params HLNeuron* hln - pointer to an HLNeuron struct
 * @params InputLayer* il - pointer to an InputLayer struct
 */
void feedHLNeuron(HLNeuron* hln, InputLayer* il) {

    hln->weightedSum = 0.0; // clear out any garbage

    // for each input[i] to HLNeuron, add il->input[i] * hln->weight[i] to HLNeuron's weighted sum
    for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
        hln->weightedSum += il->input[i] * hln->weight[i];
    }

    // apply sigmoid activation to the hln's weighted sum plus bias
    hln->output = sigmoid(hln->weightedSum + hln->bias);

} //end feedHLNeuron function

/*
 * feedNetwork - feedforwards the image then returns the int classification predicted by the network
 * @params InputLayer* il - pointer to an InputLayer struct
 * @params HiddenLayer* hl - pointer to a HiddenLayer struct
 * @params OutputLayer* ol - pointer to an OutputLayer struct
 * @params MNIST_Image* image - pointer to an MNIST_Image struct
 */
int feedNetwork(InputLayer* il, HiddenLayer* hl, OutputLayer* ol, MNIST_Image* image) {

    int prediction;

    feedInputLayer(il, image); // feed image pixel-values into input-layer
    feedHiddenLayer(hl, il);   // feed input-layer values into hidden-layer
    feedOutputLayer(ol, hl);   // feed hidden-layer values inot output-layer

    prediction = getNetworkPrediction(ol); // get the network's classification prediction

    return prediction;
} //end feedNetwork function

/*
 * feedOLNeuron - feeds hidden-layer values into an output-layer neuron
 * @params OLNeuron* oln - pointer to an OLNeuron struct
 * @params HiddenLayer* hl - pointer to a HiddenLayer struct
 */
void feedOLNeuron(OLNeuron* oln, HiddenLayer* hl) {

    oln->weightedSum = 0.0; // clear out any garbage

    // for each input[i] to OLNeuron, add hl->hNeuron[i].output * oln->weight[i] to OLNeuron's weighted sum
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        oln->weightedSum += hl->hNeuron[i].output * oln->weight[i];
    }

    // apply sigmoid activation to the oln's weighted sum plus bias
    oln->output = sigmoid(oln->weightedSum + oln->bias);

} //end feedOLNeuron function

/*
 * feedOutputLayer - feeds hidden-layer values into output-layer
 * @params OutputLayer* ol - pointer to an OutputLayer struct
 * @params HiddenLayer* hl - pointer to a HiddenLayer struct
 */
void feedOutputLayer(OutputLayer* ol, HiddenLayer* hl) {

    // for each ol->oNeuron[i], feedOLNeuron
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        feedOLNeuron(&(ol->oNeuron[i]), hl);
    }

} //end feedOutputLayer function

/*
 * getExpectedOutput - returns an ExpectedOutput struct
 * @params int mnistLabel - the label of an mnist sample
 */
ExpectedOutput getExpectedOutput(int mnistLabel) {

    ExpectedOutput expected;
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        // if i == mnistLabel set expected.value[i] to 1, else set to 0
        expected.value[i] = (i == mnistLabel ? 1 : 0);
    }
    return expected;
} //end getExpectedOutput function

/*
 * getNetworkPrediction - returns the int classification predicted by the network
 * @params OutputLayer* ol - pointer to an OutputLayer struct
 */
int getNetworkPrediction(OutputLayer* ol) {

    float highest = 0;
    int classification = 0;

    // for each oNeuron in OutputLayer
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        // if output is greater than anything we've seen so far
        if (ol->oNeuron[i].output > highest) {
            highest = ol->oNeuron[i].output;
            classification = i;
        }
    }

    return classification;
} //end getNetworkPrediction function

/*
 * initNetwork - initializes all neurons in the network with appropriate values
 * @params HiddenLayer* hl - pointer to a HiddenLayer struct
 * @params OutputLayer* ol - pointer to an OutputLayer struct
 */
void initNetwork(HiddenLayer* hl, OutputLayer* ol) {

    // seed pseudo-random number generator with current time
    srand(time(NULL));

    // for all HLNeurons in HiddenLayer
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        // for all hNeurons:
        // (A) set weight to random in range -1.0 - 1.0 inclusive
        for (int j = 0; j < INPUT_LAYER_SIZE; j++) {
            hl->hNeuron[i].weight[j] = 2 * (rand() / (float) (RAND_MAX)) - 1;
        }

        // (B) set weightedSum, bias, output, and delta to zero
        hl->hNeuron[i].weightedSum = 0.0;
        hl->hNeuron[i].bias = 0.0;
        hl->hNeuron[i].output = 0.0;
        hl->hNeuron[i].delta = 0.0;
    }

    // for all OLNeurons in OutputLayer
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        // for all oNeurons:
        // (A) set weight to random in range -1.0 - 1.0 inclusive
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
            ol->oNeuron[i].weight[j] = 2 * (rand() / (float) (RAND_MAX)) - 1;
        }

        // (B) set weightedSum, bias, output, and delta to zero
        ol->oNeuron[i].weightedSum = 0.0;
        ol->oNeuron[i].bias = 0.0;
        ol->oNeuron[i].output = 0.0;
        ol->oNeuron[i].delta = 0.0;
    }

} //end initNetwork function

/*
 * sigmoid - returns the Sigmoid activation of x
 * @params const float x - some floating-point value
 */
float sigmoid(const float x) {
    return (1.0 / (1.0 + exp((x * -1.0))));
} //end sigmoid function

/*
 * sigmoidPrime - returns the Sigmoid derivative of x
 * @params const float x - some floating-point value
 */
float sigmoidPrime(const float x) {
    return sigmoid(x) * (1.0 - sigmoid(x));
} //end sigmoidPrime function

/*
 * train - trains a neural network using the host machine
 * @params InputLayer* il - pointer to an InputLayer struct
 * @params HiddenLayer* hl - pointer to a HiddenLayer struct
 * @params OutputLayer* ol - pointer to an OutputLayer struct
 */
void train(InputLayer* il, HiddenLayer* hl, OutputLayer* ol) {

    printf("\n--- beginning training on host ---\n");

    // for each epoch
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        printf(" --- starting epoch %d of %d ---\n", epoch + 1, EPOCHS);

        // open MNIST files
        FILE* imageFile, *labelFile;
        imageFile = openMNISTImageFile(MNIST_TRAINING_SET_IMAGES_PATH);
        labelFile = openMNISTLabelFile(MNIST_TRAINING_SET_LABELS_PATH);

        // for each MNIST sample in the training set
        for (int sample = 0; sample < MNIST_TRAINING_SET_SIZE; sample++) {
            // read the next sample image and label
            MNIST_Image image = getImage(imageFile);
            MNIST_Label label = getLabel(labelFile);
            trainNetwork(il, hl, ol, &image, label);

            if ((sample + 1) % 10000 == 0) {
                printf("    => sample %d of %d complete\n", sample + 1, MNIST_TRAINING_SET_SIZE);
            }
        }

        // Close files
        fclose(imageFile);
        fclose(labelFile);
        printf(" --- epoch %d of %d complete ---\n", epoch + 1, EPOCHS);
    }

    printf("\n--- training on host complete ---\n\n");

} //end train function

/*
 * trainNetwork - performs a single feedforward, backpropagation, update weights and biases cycle
 * @params InputLayer* il - pointer to an InputLayer struct
 * @params HiddenLayer* hl - pointer to a HiddenLayer struct
 * @params OutputLayer* ol - pointer to an OutputLayer struct
 * @params MNIST_Image* image - pointer to an MNIST_Image struct
 * @params int target - the MNIST label
 */
void trainNetwork(InputLayer* il, HiddenLayer* hl, OutputLayer* ol, MNIST_Image* image, int target) {

    // (A) Feedforward Step
    feedInputLayer(il, image); // feed image pixel-values into input-layer
    feedHiddenLayer(hl, il);   // feed input-layer values into hidden-layer
    feedOutputLayer(ol, hl);   // feed hidden-layer values into output-layer

    // (B) Backpropagation Step
    ExpectedOutput expected = getExpectedOutput(target);
    calculateOutputLayerDeltas(ol, &expected);
    calculateHiddenLayerDeltas(hl, ol);

    // (C) Update Weights & Biases
    updateNetworkWeightsAndBiases(il, hl, ol);

} //end trainNetwork function

/*
 * updateHLNeuronWeightsAndBiases - updates the weights and biases of a single HLNeuron
 * @params HLNeuron* hln - pointer to an HLNeuron struct
 * @params InputLayer* il - pointer to an InputLayer struct
 */
void updateHLNeuronWeightsAndBiases(HLNeuron* hln, InputLayer* il) {

    // update each weight between InputLayer and HLNeuron
    for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
        hln->weight[i] += LEARNING_RATE * il->input[i] * hln->delta;
    }

    // update HLNeuron's bias
    hln->bias += LEARNING_RATE * hln->delta;

} //end updateHLNeuronWeightsAndBiases function

/*
 * updateNetworkWeightsAndBiases - updates all weights and biases in the network
 * @params InputLayer* il - pointer to an InputLayer struct
 * @params HiddenLayer* hl - pointer to a HiddenLayer struct
 * @params OutputLayer* ol - pointer to an OutputLayer struct
 */
void updateNetworkWeightsAndBiases(InputLayer* il, HiddenLayer* hl, OutputLayer* ol) {

    // for each hNeuron in HiddenLayer
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        updateHLNeuronWeightsAndBiases(&(hl->hNeuron[i]), il);
    }

    // for each oNeuron in OutputLayer
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        updateOLNeuronWeightsAndBiases(&(ol->oNeuron[i]), hl);
    }

} //end updateNetworkWeightsAndBiases function

/*
 * updateOLNeuronWeightsAndBiases - updates the weights and biases of a single OLNeuron
 * @params OLNeuron* oln - pointer to an OLNeuron struct
 * @params HiddenLayer* hl - pointer to a HiddenLayer struct
 */
void updateOLNeuronWeightsAndBiases(OLNeuron* oln, HiddenLayer* hl) {

    // update each weight between HiddenLayer and OLNeuron
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        oln->weight[i] += LEARNING_RATE * hl->hNeuron[i].output * oln->delta;
    }

    // update OLNeuron's bias
    oln->bias += LEARNING_RATE * oln->delta;

} //end updateOLNeuronWeightsAndBiases function
