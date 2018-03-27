/*******************************************************************************************
 * Filename: ui_train.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the user interface for training a Neuronmancer model.
 *******************************************************************************************
 */

/* ui_train method - user interface for training a model */
void ui_train() {
    // declare variables needed to store the model information
    char inputBuffer[MAXINPUT]; // store the user's input (gets recycled a lot)
    int tempInt; // store temp int input from user (used for determining whether to run on host or GPU device)
    int numberOfLayers; // store the total number of layers in the network
    int numberOfNeuronsTotal; // store the total number of neurons in our neural network
    int numberOfWeightsTotal; // store the total number of weights in our neural network
    int* numberOfNeuronsPerLayer; // store the total number of neurons in each layer in our neural network in a 1d array of size numberOfLayers
    int* numberOfWeightsPerLayer; // store the total number of weights between each layer in our neural network in a 1d array of size numberOfLayers-1
    int* firstNeuronIndexPerLayer; // store the indexes of each layer's first neuron value
    int* firstWeightIndexPerLayer; // store the indexes of each layer's first weight value
    Activation* activationsPerLayer; // store the activation of each layer
    double* weights; // store the weight values of our neural network in a 1d array of size weightSize (1d arrays are easy to work with in CUDA)
    double* biases; // store the biases values of our neural network in a 1d array of size weightSize (1d arrays are easy to work with in CUDA)
    int epochs; // store the number of epochs for training
    double learningRate; // store the rate that our network will learn

    int myPatience = 2; // stores the amount of patience I have for the user's nonsense

    // initialize pointers with malloc (will be resized in readmodel.cu later)
    numberOfNeuronsPerLayer = (int *) malloc(sizeof(int));
    if (numberOfNeuronsPerLayer == NULL) {
        onMallocError(sizeof(int));
    }

    numberOfWeightsPerLayer = (int *) malloc(sizeof(int));
    if (numberOfWeightsPerLayer == NULL) {
        onMallocError(sizeof(int));
    }

    firstNeuronIndexPerLayer = (int *) malloc(sizeof(int));
    if (firstNeuronIndexPerLayer == NULL) {
        onMallocError(sizeof(int));
    }

    firstWeightIndexPerLayer = (int *) malloc(sizeof(int));
    if (firstWeightIndexPerLayer == NULL) {
        onMallocError(sizeof(int));
    }

    activationsPerLayer = (Activation *) malloc(sizeof(Activation));
    if (activationsPerLayer == NULL) {
        onMallocError(sizeof(Activation));
    }

    weights = (double *) malloc(sizeof(double));
    if (weights == NULL) {
        onMallocError(sizeof(double));
    }

    biases = (double *) malloc(sizeof(double));
    if (biases == NULL) {
        onMallocError(sizeof(double));
    }

    printf("Lets train an artificial neural network!\n");
    printf("Searching ./nmModel for files...\n");

    readModel(&numberOfLayers, &numberOfNeuronsTotal, &numberOfWeightsTotal, numberOfNeuronsPerLayer, numberOfWeightsPerLayer, firstNeuronIndexPerLayer,
            firstWeightIndexPerLayer, weights, biases, &learningRate, &epochs);

    printf("...files found!\n");

#ifdef DEBUG
    printf("epochs                 = %d\n", epochs);
    printf("learningRate           = %lf\n", learningRate);
    printf("numberOfLayers         = %d\n", numberOfLayers);
    printf("numberOfNeuronsTotal   = %d\n", numberOfNeuronsTotal); // remember, numberOfNeuronsTotal equals numberOfBiasesTotal
    printf("numberOfWeightsTotal   = %d\n", numberOfWeightsTotal);

    for(int i = 0; i < numberOfLayers; i++) {
        printf("numberOfNeuronsPerLayer[%d]  = %d\n", i, numberOfNeuronsPerLayer[i]);
        printf("numberOfWeightsPerLayer[%d]  = %d\n", i, numberOfWeightsPerLayer[i]);
        printf("firstNeuronIndexPerLayer[%d] = %d\n", i, firstNeuronIndexPerLayer[i]);
        printf("firstWeightIndexPerLayer[%d] = %d\n", i, firstWeightIndexPerLayer[i]);
        if(activationsPerLayer[i] == SIGMACT) {
            printf("activationsPerLayer[%d] = SIGMOID\n", i);
        } else if(activationsPerLayer[i] == RELUACT) {
            printf("activationsPerLayer[%d] = RELU\n", i);
        } else if(activationsPerLayer[i] == TANHACT) {
            printf("activationsPerLayer[%d] = TANH\n", i);
        } else {
            printf("activationsPerLayer[%d] = %d\n", i, activationsPerLayer[i]);
        }
    }

    printarray("biases", biases, numberOfNeuronsTotal);
    printarray("weights", weights, numberOfWeightsTotal);
#endif

    // get user input for running on CPU or GPU
    tempInt = 'z'; // assign 'z' to enter loop
    while (tempInt != 'h' || tempInt != 'H' || tempInt != 'd' || tempInt != 'D') {
        // get the activation for layer i
        printf("Do you want to train on the host machine or GPU device?\nEnter h for host or d for device:\n~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
        sscanf(inputBuffer, "%d", &tempInt); // format and dump the user's input
        if (tempInt != 'h' || tempInt != 'H' || tempInt != 'd' || tempInt != 'D') {
            onInvalidInput(myPatience);
            myPatience--;
        }
    }
    myPatience = 2; // restore my patience

    if (tempInt != 'h' || tempInt != 'H') {
        printf("Today we keep tradition, looks like we're training on the host machine!\n");

        // TODO: START HOST TRAINING
        printf("Beginning training now...");

        // TODO: do(LOADINPUT, FEEDFORWARD, COMPAREOUTPUT, BACKPROPAGATEERRS) for all samples in batch, WEIGHTUPDATE & BIASUPDATE, then repeat until i == epochs
        for (int i = 0; i < epochs; i++) {
            //printf("Starting load input step now...\n");
            loadInput(neurons, numberOfNeuronsPerLayer[0]); // load some random input for feedforward testing
            //printarray("neurons", neurons, numberOfNeuronsTotal);

            //printf("Starting feedforward step now...\n");
            feedforwardWithHost(neurons, weights, numberOfLayers, numberOfNeuronsPerLayer, numberOfWeightsPerLayer, firstNeuronIndexPerLayer,
                    firstWeightIndexPerLayer); // feed the input forward

            //printf("Network state post feedforward:\n");
            //printarray("neurons", neurons, numberOfNeuronsTotal);
            //printarray("weights", weights, numberOfWeightsTotal);

            //printf("Generating random training labels for testing backpropagation now...\n");
            loadInput(outputExpected, numberOfNeuronsPerLayer[numberOfLayers - 1]); // load some random input for backpropagation testing
            //printarray("outputExpected", outputExpected, numberOfNeuronsPerLayer[numberOfLayers - 1]);

            //printf("Starting backpropagation step now...\n");
            backpropagateWithHost(outputExpected, neurons, weights, neuronErrors, numberOfLayers, numberOfNeuronsPerLayer, numberOfWeightsPerLayer,
                    firstNeuronIndexPerLayer, firstWeightIndexPerLayer, learningRate); // calculate and back propagate errors

            //printf("Network state post backpropagation:\n");
            //printarray("neurons", neurons, numberOfNeuronsTotal);
            //printarray("weights", weights, numberOfWeightsTotal);
            if(i % 10 == 0) {
                printf("...%d epochs complete...", i);
            }

        }


    } else if (tempInt != 'd' || tempInt != 'D') {
        printf("Today we break with tradition, looks like we're training on the GPU device!\n");
        // declare our device variables
        int* deviceNumberOfNeuronsPerLayer;
        int* deviceNumberOfWeightsPerLayer;
        double* deviceNeurons;
        double* deviceWeights;
        double* deviceWeightCosts;

        // declare our cudaStatus variable
        cudaError_t cudaStatus;

        // run on GPU 0, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
	    if (cudaStatus != cudaSuccess) {
            onFailToSetGPUDevice();
        }

        printf("Allocating GPU device memory and copying host values over...\n");

        // allocate device memory for device variables and copy host values to device copies
        cudaStatus = cudaMalloc((void **) &deviceNumberOfNeuronsPerLayer, numberOfLayers * sizeof(int)); //cudaMalloc allocates a chunk of device memory
	    if (cudaStatus != cudaSuccess) {
            onCudaMallocError(numberOfLayers * sizeof(int);
        }

        cudaStatus = cudaMalloc((void **) &deviceNumberOfWeightsPerLayer, numberOfLayers * sizeof(int)); //cudaMalloc allocates a chunk of device memory
	    if (cudaStatus != cudaSuccess) {
            onCudaMallocError(numberOfLayers * sizeof(int);
        }

        cudaStatus = cudaMalloc((void **) &deviceNeurons, (numberOfNeuronsTotal * sizeof(double))); //cudaMalloc allocates a chunk of device memory
	    if (cudaStatus != cudaSuccess) {
            onCudaMallocError(numberOfNeuronsTotal * sizeof(double);
        }

        cudaStatus = cudaMalloc((void **) &deviceWeights, (numberOfWeightsTotal * sizeof(double))); //cudaMalloc allocates a chunk of device memory
	    if (cudaStatus != cudaSuccess) {
            onCudaMallocError(numberOfWeightsTotal * sizeof(double);
        }

        cudaStatus = cudaMalloc((void **) &deviceWeightCosts, (numberOfWeightsTotal * sizeof(double))); //cudaMalloc allocates a chunk of device memory
	    if (cudaStatus != cudaSuccess) {
            onCudaMallocError(numberOfWeightsTotal * sizeof(double);
        }

        cudaStatus = cudaMemcpy(deviceNumberOfNeuronsPerLayer, numberOfNeuronsPerLayer, (numberOfLayers * sizeof(int)), cudaMemcpyHostToDevice); //cudaMemcpy copies host values to device copies
	    if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("numberOfNeuronsPerLayer");
        }

        cudaStatus = cudaMemcpy(deviceNumberOfWeightsPerLayer, numberOfWeightsPerLayer, (numberOfLayers * sizeof(int)), cudaMemcpyHostToDevice); //cudaMemcpy copies host values to device copies
	    if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("numberOfWeightsPerLayer");
        }

        cudaStatus = cudaMemcpy(deviceNeurons, neurons, (numberOfNeuronsTotal * sizeof(double)), cudaMemcpyHostToDevice); //cudaMemcpy copies host values to device copies
	    if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("numberOfNeuronsTotal");
        }

        cudaStatus = cudaMemcpy(deviceWeights, weights, (numberOfWeightsTotal * sizeof(double)), cudaMemcpyHostToDevice); //cudaMemcpy copies host values to device copies
	    if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("numberOfWeightsTotal");
        }
        printf("...allocation successful!\n");

        // TODO: START GPU DEVICE TRAINING
        // TODO: do(LOADINPUT, FEEDFORWARD, COMPAREOUTPUT, BACKPROPAGATEERRS) for all samples in batch, WEIGHTUPDATE & BIASUPDATE, then repeat until i == epochs
        

        // TODO: COPY DEVICE VARIABLE VALUES BACK TO HOST

        printf("Press enter to free dynamically allocated GPU device memory.\n~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input

        printf("Freeing dynamically allocated GPU device memory...");

        // free the chunks of device memory that were dynamically allocated by cudaMalloc
        cudaFree(deviceNumberOfNeuronsPerLayer);
        cudaFree(deviceNumberOfWeightsPerLayer);
        cudaFree(deviceNeurons);
        cudaFree(deviceWeights);
        cudaFree(deviceWeightCosts);
    }

    // TODO: SAVE TRAINED WEIGHTS AND BIASES TO DISK

    printf("Press enter to free dynamically allocated host memory.\n~");
    fgets(inputBuffer, MAXINPUT, stdin); // read the user's input

    printf("Freeing dynamically allocated host memory...");

    // free the chunks of host memory that were dynamically allocated by malloc
    free(numberOfNeuronsPerLayer);
    free(numberOfWeightsPerLayer);
    free(firstNeuronIndexPerLayer);
    free(firstWeightIndexPerLayer);
    free(neurons);
    free(weights);
    free(neuronErrors);
    free(outputExpected);

    printf("Press enter to return to the main menu:\n~");
    fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
    printf("\n");
} //end ui_train method
