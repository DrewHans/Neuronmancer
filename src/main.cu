/********************************************************************************
 * Filename: main.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: main function for Neuronmancer
 ********************************************************************************
 */

/* main - the program starts here
 * @params: argc - the argument count
 * @params: argv - the string of arguments
 */
int main(int argc, char * argv[]) {
    printf("Starting Neuronmancer...\n");

    char buffer[MAXINPUT]; // store the user's input (gets recycled a lot)
    int myPatience = 2; // stores the amount of patience I have for the user's nonsense
    int loop = -1;

    while (loop != 0) {
        printf("What would you like to do?\nEnter 0 to quit, 1 to create a new model, 2 to train an existing model, or 3 to evaluate an existing model:\n~");
        fgets(buffer, MAXINPUT, stdin); // read the user's input from stdin into inputBuffer
        sscanf(buffer, "%d", &loop); // format and dump the user's input into loop
        if (loop < 0 || loop > 3) {
            onInvalidInput(myPatience);
            myPatience--;
        } else if (loop == 1) {
            printf("\nLets create an artificial neural network!\n");
            ui_create(); // jump into user-interface for creating an ANN model
            myPatience = 2; // restore my patience
        } else if (loop == 2) {
            printf("\nLets train an artificial neural network!\n");
            ui_train(); // jump into user-interface for training an existing ANN model
            myPatience = 2; // restore my patience
        } else if (loop == 3) {
            printf("\nLets evaluate an artificial neural network!\n");
            ui_evaluate(); // jump into user-interface for evaluating an existing ANN model
            myPatience = 2; // restore my patience
        }
    }//end while loop

    printf("Neuronmancer will now end.\n");
} //end main function

