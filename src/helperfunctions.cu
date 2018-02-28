/*******************************************************************************************
 * Filename: helperfunctions.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains miscellaneous helper functions.
 *******************************************************************************************
 */

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

