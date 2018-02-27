/*******************************************************************************************
 * Filename: feedforwardfunctions.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the host and device cost functions.
 *******************************************************************************************
 */

/*
 * costFunction method - compares a calculated output value to the outputExpected value and returns the error amount
 * @params: expectedValue - a pointer to a double value
 * @params: calculatedValue - a pointer to a double value
 * @returns: the difference between outputExpected and calculated values
 */
double costFunction(double* expectedValue, double* calculatedValue) {
    return expectedValue - calculatedValue;
}//end costFunction method
