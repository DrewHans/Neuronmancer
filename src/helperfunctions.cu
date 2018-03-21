/*******************************************************************************************
 * Filename: helperfunctions.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains helper functions - simple array operations, dealing 
                with critical errors, printing insults, etc.
 *******************************************************************************************
 */
 
/*
 * initArrayToRandomDoubles method
 * @params: a - a pointer to an array of double values
 * @params: n - the size of array a
 */
void initArrayToRandomDoubles(double* a, int n) {
    // generate random doubles in range [0, 1)
    for (int i = 0; i < n; i++) {
        a[i] = ((double) rand()) / ((double) RAND_MAX);
    }
}//end initArrayToRandomDoubles method

/*
 * initArrayToZeros method
 * @params: a - a pointer to an array of double values
 * @params: n - the size of array a
 */
void initArrayToZeros(double* a, int n) {
    // set all neuron values to zero
    for (int i = 0; i < n; i++) {
        a[i] = 0;
    }
}//end initArrayToZeros method

/*
 * printarray method - prints out array values to terminal
 * @params: name - a pointer to a char string
 * @params: array - a pointer to an array of double values
 * @params: n - the size of array
 */
void printarray(const char* name, double* array, int n) {
    for (int i = 0; i < n; i++) {
        printf("%s[%d]=%f\n", name, i, array[i]);
    }
    printf("\n");
}//end printarray method

/*
 * printFarewellMSG method - prints out one final insult
 */
void printFarewellMSG() {
    printf("Sorry, I did everything I could but it looks like I'm crashing...\n...\n...your computer sucks, good-bye.\n");
}//end printFarewellMSG method

/*
 * onFileOpenError method - SOS, we're going down
 * @params: path - file that failed to open
 */
void onFileOpenError(const char* path) {
    printf("ERROR: Failed to open file %s!\n", path);
    printFarewellMSG();
    exit(1);
}//end onFileOpenError method

/*
 * onInvalidInput method - prints out insults when the user screws up (silly humans)
 * @params: myPatience - the current state of my patience, represented as an int
 */
void onInvalidInput(int myPatience) {
    if(myPatience == 2) {
        printf("Looks like you entered an illegal value... you're testing my patience, try again!\n");
    } else if(myPatience == 1) {
        printf("That's the second time you've entered an illegal value... do you think this is funny? Try again!\n");
    } else if(myPatience == 0) {
        printf("Sigh... you just can't do anything right, can you?\n");
    } else {
        printf("Look dude, I've got all day. If you wanna keep wasting your time then that's fine by me. You know what you're supposed to do.\n");
    }
}//end onInvalidInput method

/*
 * onMallocError method - SOS, we're going down
 * @params: size - the size of the memory that we couldn't allocate
 */
void onMallocError(int size) {
    printf("ERROR: Failed to malloc %d of memory!\n", size);
    printFarewellMSG();
    exit(1);
}//end onMallocError method
