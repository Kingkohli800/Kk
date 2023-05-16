Q.1 Write a program to implement Fuzzy Operations 
Union
Program:
#include <stdio.h>
#define MAX_SIZE 100
void fuzzyUnion(double A[], double B[], double result[], int size) { 
for (int i = 0; i < size; i++) {
 result[i] = A[i] > B[i] ? A[i] : B[i];
 }
}
void printFuzzySet(double A[], int size) { 
for (int i = 0; i < size; i++) {
 printf("%.2f ", A[i]);
 }
 printf("\n");
}
int main() {
 double setA[MAX_SIZE], setB[MAX_SIZE], result[MAX_SIZE]; 
int size;
 printf("Enter the size of fuzzy sets (up to %d): ", MAX_SIZE); 
scanf("%d", &size);
 printf("Enter elements of set A:\n");
 for (int i = 0; i < size; i++) {
 scanf("%lf", &setA[i]);
 }
 printf("Enter elements of set B:\n");
 for (int i = 0; i < size; i++) {
 scanf("%lf", &setB[i]);
 }
 printf("Fuzzy Set A: "); 
printFuzzySet(setA, size); printf("Fuzzy 
Set B: ");
 printFuzzySet(setB, size);
 fuzzyUnion(setA, setB, result, size);
 printf("Fuzzy Union: ");
 printFuzzySet(result, size);
} 
Output:
C:\Users\rutuja mundada\Desktop>myexe
Enter the size of fuzzy sets (up to 100): 10 Enter 
elements of set A:
2
3
5
8
5
9
3
7
1
5
Enter elements of set B:
4
7
2
4
6
8
1
2
4
7
Fuzzy Set A: 2.00 3.00 5.00 8.00 5.00 9.00 3.00 7.00 1.00 5.00
Fuzzy Set B: 4.00 7.00 2.00 4.00 6.00 8.00 1.00 2.00 4.00 7.00
Fuzzy Union: 4.00 7.00 5.00 8.00 6.00 9.00 3.00 7.00 4.00 7.00





Intersection:
#include <stdio.h>
#define MAX_SIZE 100
void fuzzyIntersection(double A[], double B[], double result[], int size) { 
for (int i = 0; i < size; i++) {
 result[i] = A[i] < B[i] ? A[i] : B[i];
 }
}
void printFuzzySet(double A[], int size) { 
for (int i = 0; i < size; i++) {
 printf("%.2f ", A[i]);
 }
 printf("\n");
}
int main() {
 double setA[MAX_SIZE], setB[MAX_SIZE], result[MAX_SIZE]; 
int size;
 printf("Enter the size of fuzzy sets (up to %d): ", MAX_SIZE); 
scanf("%d", &size);
 printf("Enter elements of set A:\n");
 for (int i = 0; i < size; i++) {
 scanf("%lf", &setA[i]);
 }
 printf("Enter elements of set B:\n");
 for (int i = 0; i < size; i++) {
 scanf("%lf", &setB[i]);
 }
 printf("Fuzzy Set A: "); 
printFuzzySet(setA, size); printf("Fuzzy 
Set B: ");
 printFuzzySet(setB, size);
 fuzzyIntersection(setA, setB, result, size); 
printf("Fuzzy Intersection: ");
 printFuzzySet(result, size);
} 
Output:
C:\Users\rutuja mundada\Desktop>myexe
Enter the size of fuzzy sets (up to 100): 10 Enter 
elements of set A:
2
3
5
8
5
9
3
7
1
5
Enter elements of set B:
4
7
2
4
6
8
1
2
4
7
Fuzzy Set A: 2.00 3.00 5.00 8.00 5.00 9.00 3.00 7.00 1.00 5.00
Fuzzy Set B: 4.00 7.00 2.00 4.00 6.00 8.00 1.00 2.00 4.00 7.00
Fuzzy Intersection: 2.00 3.00 2.00 4.00 5.00 8.00 1.00 2.00 1.00 5.00







Complement:
#include <stdio.h>
#define MAX_SIZE 100
void fuzzyComplement(double A[], double result[], int size) { 
for (int i = 0; i < size; i++) {
 result[i] = 1.0 - A[i];
 }
}
void printFuzzySet(double A[], int size) { 
for (int i = 0; i < size; i++) {
 printf("%.2f ", A[i]);
 }
 printf("\n");
}
int main() {
 double setA[MAX_SIZE], setB[MAX_SIZE], result[MAX_SIZE]; 
int size;
 printf("Enter the size of fuzzy sets (up to %d): ", MAX_SIZE); 
scanf("%d", &size);
 printf("Enter elements of set A:\n");
 for (int i = 0; i < size; i++) {
 scanf("%lf", &setA[i]);
 }
 printf("Enter elements of set B:\n");
 for (int i = 0; i < size; i++) {
 scanf("%lf", &setB[i]);
 }
 printf("Fuzzy Set A: "); 
printFuzzySet(setA, size); printf("Fuzzy 
Set B: ");
 printFuzzySet(setB, size);
 fuzzyComplement(setA, result, size); 
printf("Fuzzy Complement of A: ");
 printFuzzySet(result, size);
}
 
Output:
C:\Users\rutuja mundada\Desktop>myexe
Enter the size of fuzzy sets (up to 100): 10 Enter 
elements of set A:
2
3
5
8
5
9
3
7
1
5
Enter elements of set B:
4
7
2
4
6
8
1
2
4
7
Fuzzy Set A: 2.00 3.00 5.00 8.00 5.00 9.00 3.00 7.00 1.00 5.00
Fuzzy Set B: 4.00 7.00 2.00 4.00 6.00 8.00 1.00 2.00 4.00 7.00
Fuzzy Complement of A: -1.00 -2.00 -4.00 -7.00 -4.00 -8.00 -2.00 -6.00 0.00 -4.00









Algebric Sum:
#include <stdio.h>
#define MAX_SIZE 100
void fuzzyAlgebraicSum(double A[], double B[], double result[], int size) { 
for (int i = 0; i < size; i++) {
 result[i] = A[i] + B[i] - A[i] * B[i];
 }
}
void printFuzzySet(double A[], int size) { 
for (int i = 0; i < size; i++) {
 printf("%.2f ", A[i]);
 }
 printf("\n");
}
int main() {
 double setA[MAX_SIZE], setB[MAX_SIZE], result[MAX_SIZE]; 
int size;
 printf("Enter the size of fuzzy sets (up to %d): ", MAX_SIZE); 
scanf("%d", &size);
 printf("Enter elements of set A:\n");
 for (int i = 0; i < size; i++) {
 scanf("%lf", &setA[i]);
 }
 printf("Enter elements of set B:\n");
 for (int i = 0; i < size; i++) {
 scanf("%lf", &setB[i]);
 }
 printf("Fuzzy Set A: "); 
printFuzzySet(setA, size); printf("Fuzzy 
Set B: ");
 printFuzzySet(setB, size);
 fuzzyAlgebraicSum(setA, setB, result, size); 
printf("Fuzzy Algebraic Sum: ");
 printFuzzySet(result, size);
} 
Output:
C:\Users\rutuja mundada\Desktop>myexe
Enter the size of fuzzy sets (up to 100): 10 Enter 
elements of set A:
2
3
5
8
5
9
3
7
1
5
Enter elements of set B:
4
7
2
4
6
8
1
2
4
7
Fuzzy Set A: 2.00 3.00 5.00 8.00 5.00 9.00 3.00 7.00 1.00 5.00
Fuzzy Set B: 4.00 7.00 2.00 4.00 6.00 8.00 1.00 2.00 4.00 7.00
Fuzzy Algebraic Sum: -2.00 -11.00 -3.00 -20.00 -19.00 -55.00 1.00 -5.00 1.00 -23.00











Algebric Product:
#include <stdio.h>
#define MAX_SIZE 100
void fuzzyAlgebraicProduct(double A[], double B[], double result[], int size) { for (int i = 0; i 
< size; i++) {
 result[i] = A[i] * B[i];
 }
}
void printFuzzySet(double A[], int size) { 
for (int i = 0; i < size; i++) {
 printf("%.2f ", A[i]);
 }
 printf("\n");
}
int main() {
 double setA[MAX_SIZE], setB[MAX_SIZE], result[MAX_SIZE]; 
int size;
 printf("Enter the size of fuzzy sets (up to %d): ", MAX_SIZE); 
scanf("%d", &size);
 printf("Enter elements of set A:\n");
 for (int i = 0; i < size; i++) {
 scanf("%lf", &setA[i]);
 }
 printf("Enter elements of set B:\n");
 for (int i = 0; i < size; i++) {
 scanf("%lf", &setB[i]);
 }
 printf("Fuzzy Set A: "); 
printFuzzySet(setA, size); printf("Fuzzy 
Set B: ");
 printFuzzySet(setB, size);
 fuzzyAlgebraicProduct(setA, setB, result, size); 
printf("Fuzzy Algebraic Product: ");
 printFuzzySet(result, size);
} 
Output:
C:\Users\rutuja mundada\Desktop>myexe
Enter the size of fuzzy sets (up to 100): 10 Enter 
elements of set A:
2
3
5
8
5
9
3
7
1
5
Enter elements of set B:
4
7
2
4
6
8
1
2
4
7
Fuzzy Set A: 2.00 3.00 5.00 8.00 5.00 9.00 3.00 7.00 1.00 5.00
Fuzzy Set B: 4.00 7.00 2.00 4.00 6.00 8.00 1.00 2.00 4.00 7.00
Fuzzy Algebraic Product: 8.00 21.00 10.00 32.00 30.00 72.00 3.00 14.00 4.00 35.00










Cartesian Product:
#include <stdio.h>
#define MAX_SIZE 100
void fuzzyCartesianProduct(double A[], double B[], double result[], int size) { 
int index = 0; for (int i = 0; i < size; i++) { for (int j = 0; j < size; j++) {
 result[index++] = A[i] * B[j];
 }
 }
}
void printFuzzySet(double A[], int size) { 
for (int i = 0; i < size; i++) {
 printf("%.2f ", A[i]);
 }
 printf("\n");
}
int main() {
 double setA[MAX_SIZE], setB[MAX_SIZE], result[MAX_SIZE]; 
int size;
 printf("Enter the size of fuzzy sets (up to %d): ", MAX_SIZE); 
scanf("%d", &size);
 printf("Enter elements of set A:\n");
 for (int i = 0; i < size; i++) {
 scanf("%lf", &setA[i]);
 }
 printf("Enter elements of set B:\n");
 for (int i = 0; i < size; i++) {
 scanf("%lf", &setB[i]);
 }
 printf("Fuzzy Set A: "); 
printFuzzySet(setA, size); printf("Fuzzy 
Set B: ");
 printFuzzySet(setB, size);
 fuzzyCartesianProduct(setA, setB, result, size); 
printf("Fuzzy Cartesian Product: ");
 printFuzzySet(result, size);
} 
Output:
C:\Users\rutuja mundada\Desktop>myexe
Enter the size of fuzzy sets (up to 100): 10 Enter 
elements of set A:
2
3
5
8
5
9
3
7
1
5
Enter elements of set B:
4
7
2
4
6
8
1
2
4
7
Fuzzy Set A: 2.00 3.00 5.00 8.00 5.00 9.00 3.00 7.00 1.00 5.00
Fuzzy Set B: 4.00 7.00 2.00 4.00 6.00 8.00 1.00 2.00 4.00 7.00
Fuzzy Cartesian Product: 8.00 14.00 4.00 8.00 12.00 16.00 2.00 4.00 8.00 14.00







Q.2 Write a program to implement De Morgans law
Program :
#include <stdio.h>
int main() { 
int x, y;
 printf("Enter the values of x and y (0 or 1):\n"); 
scanf("%d %d", &x, &y);
 // Applying De Morgan's Law 
int notXandY = !(x && y);
 int notXorNotY = (!x) || (!y);
 printf("Applying De Morgan's Law:\n"); 
printf("NOT(x AND y) = %d\n", notXandY);
 printf("NOT(x) OR NOT(y) = %d\n", notXorNotY);
 return 0;
}
Output
C:\Users\rutuja mundada\Desktop>myexe Enter 
the values of x and y (0 or 1):
1
0
Applying De Morgan's Law:
NOT(x AND y) = 1
NOT(x) OR NOT(y) = 1







Q.3 Write a program to implement Max-Min Composition and Max-Product 
Composition.
Program:
#include <stdio.h>
#define MAX_ROWS 10
#define MAX_COLS 10
void maxMinComposition(int A[MAX_ROWS][MAX_COLS], int B[MAX_ROWS][MAX_COLS], 
int result[MAX_ROWS][MAX_COLS], int rows, int cols) {
 for (int i = 0; i < rows; i++) { 
for (int j = 0; j < cols; j++) {
 int max = A[i][0] > B[0][j] ? A[i][0] : B[0][j]; 
for (int k = 1; k < cols; k++) {
 int temp = A[i][k] > B[k][j] ? A[i][k] : B[k][j];
 max = max < temp ? max : temp;
 }
 result[i][j] = max;
 }
 }
}
void maxProductComposition(int A[MAX_ROWS][MAX_COLS], int 
B[MAX_ROWS][MAX_COLS], int result[MAX_ROWS][MAX_COLS], int rows, int cols) { 
for (int i = 0; i < rows; i++) { for (int j = 0; j < cols; j++) { int maxProduct = 
A[i][0] * B[0][j]; for (int k = 1; k < cols; k++) {
 int temp = A[i][k] * B[k][j];
 maxProduct = maxProduct > temp ? maxProduct : temp;
 }
 result[i][j] = maxProduct;
 }
 }
}
void printMatrix(int matrix[MAX_ROWS][MAX_COLS], int rows, int cols) { 
for (int i = 0; i < rows; i++) { for (int j = 0; j < cols; j++) {
 printf("%d ", matrix[i][j]);
 }
 printf("\n");
 }
}
int main() {
 int matrixA[MAX_ROWS][MAX_COLS]; int 
matrixB[MAX_ROWS][MAX_COLS]; int 
resultMaxMin[MAX_ROWS][MAX_COLS];
 int resultMaxProduct[MAX_ROWS][MAX_COLS]; 
int rows, cols;
 printf("Enter the number of rows (up to %d): ", MAX_ROWS); 
scanf("%d", &rows);
 printf("Enter the number of columns (up to %d): ", MAX_COLS); 
scanf("%d", &cols);
 printf("Enter elements of matrix A:\n");
 for (int i = 0; i < rows; i++) { 
for (int j = 0; j < cols; j++) {
 scanf("%d", &matrixA[i][j]);
 }
 }
 printf("Enter elements of matrix B:\n");
 for (int i = 0; i < rows; i++) { 
for (int j = 0; j < cols; j++) {
 scanf("%d", &matrixB[i][j]);
 }
 }
 maxMinComposition(matrixA, matrixB, resultMaxMin, rows, cols); 
maxProductComposition(matrixA, matrixB, resultMaxProduct, rows, cols);
 printf("Max-Min Composition:\n");
 printMatrix(resultMaxMin, rows, cols);
 printf("\nMax-Product Composition:\n");
 printMatrix(resultMaxProduct, rows, cols);
 return 0;
}
Output
C:\Users\rutuja mundada\Desktop>myexe
Enter the number of rows (up to 10): 3 Enter 
the number of columns (up to 10): 3 Enter 
elements of matrix A:
2
3
4 
5
6
7
8
9
2
Enter elements of matrix B:
1
2
3
44
22
33
65
13
47
Max-Min Composition:
2 2 3
5 5 5
8 8 8
Max-Product Composition:
260 66 188
455 132 329
396 198 297







Q.4 Write a program to implement lambda cut
Program:
#include <stdio.h>
#define MAX_SIZE 100
// Function to apply Lambda Cut to a fuzzy set void 
lambdaCut(float set[], int size, float lambda) {
 printf("Lambda Cut (λ = %.2f): ", lambda); 
for (int i = 0; i < size; i++) { if (set[i] >= 
lambda) { printf("1 "); } else { 
printf("0 ");
 }
 }
 printf("\n");
}
int main() { float 
fuzzySet[MAX_SIZE];
 int size;
 float lambda;
 printf("Enter the number of elements in the fuzzy set: "); 
scanf("%d", &size);
 printf("Enter the elements of the fuzzy set (separated by spaces): "); 
for (int i = 0; i < size; i++) {
 scanf("%f", &fuzzySet[i]);
 }
 printf("Enter the value of lambda: "); 
scanf("%f", &lambda);
 lambdaCut(fuzzySet, size, lambda);
 return 0;
} 
Output:
C:\Users\rutuja mundada\Desktop>myexe
Enter the number of elements in the fuzzy set: 4
Enter the elements of the fuzzy set (separated by spaces): 2
5
9
2
Enter the value of lambda: 2
Lambda Cut (╬╗ = 2.00): 1 1 1 1







Q.5 Write a program to implement Activation Function
Program:
#include <stdio.h>
#include <math.h>
// Function to calculate the sigmoid activation double 
sigmoid(double x) {
 return 1 / (1 + exp(-x));
}
int main() {
 double input;
 printf("Enter the input value: ");
 scanf("%lf", &input);
 double result = sigmoid(input);
 printf("Sigmoid Activation: %.4lf\n", result);
 return 0;
}
Output:
C:\Users\rutuja mundada\Desktop>myexe
Enter the input value: 5
Sigmoid Activation: 0.9933




















Q.6 Write a program to implement Perceptron Learning Rule
Program:
#include <stdio.h>
#define MAX_SIZE 100
// Function to calculate the dot product of two vectors double 
dotProduct(double inputs[], double weights[], int size) { 
double result = 0.0; for (int i = 0; i < size; i++) {
 result += inputs[i] * weights[i];
 }
 return result;
}
// Function to apply the Perceptron Learning Rule
void perceptronLearningRule(double inputs[][MAX_SIZE], double outputs[], double 
weights[], int numSamples, int numInputs, double learningRate, int maxIterations) { 
int iterations = 0;
 int error;
 do { 
error = 0;
 for (int i = 0; i < numSamples; i++) { 
double inputVector[MAX_SIZE]; for 
(int j = 0; j < numInputs; j++) {
 inputVector[j] = inputs[i][j];
 }
 double predictedOutput = dotProduct(inputVector, weights, numInputs); 
int predictedLabel = (predictedOutput > 0) ? 1 : 0;
 int actualLabel = outputs[i];
 if (predictedLabel != actualLabel) { 
for (int j = 0; j < numInputs; j++) {
 weights[j] += learningRate * (actualLabel - predictedLabel) * inputVector[j];
 }
 error = 1;
 }
 }
 iterations++;
 } while (error && iterations < maxIterations);
 if (iterations == maxIterations) {
 printf("Perceptron Learning Rule did not converge within the maximum number of 
iterations.\n");
 } else {
 printf("Perceptron Learning Rule converged in %d iterations.\n", iterations); 
printf("Weights: "); for (int i = 0; i < numInputs; i++) {
 printf("%.2lf ", weights[i]);
 }
 printf("\n");
 }
}
int main() {
 int numSamples, numInputs;
 printf("Enter the number of samples: ");
 scanf("%d", &numSamples);
 printf("Enter the number of inputs: ");
 scanf("%d", &numInputs);
 double inputs[MAX_SIZE][MAX_SIZE]; 
double outputs[MAX_SIZE];
 double weights[MAX_SIZE] = {0.0};
 printf("Enter the input values for each sample:\n");
 for (int i = 0; i < numSamples; i++) { 
printf("Sample %d:\n", i + 1); for 
(int j = 0; j < numInputs; j++) {
 scanf("%lf", &inputs[i][j]);
 }
 }
 printf("Enter the output values for each sample:\n");
 for (int i = 0; i < numSamples; i++) { 
printf("Sample %d: ", i + 1); scanf("%lf", 
&outputs[i]);
 }
 double learningRate;
 int maxIterations;
 printf("Enter the learning rate: "); 
scanf("%lf", &learningRate);
 printf("Enter the maximum number of iterations: "); 
scanf("%d", &maxIterations);
 perceptronLearningRule(inputs, outputs, weights, numSamples, numInputs, learningRate, 
maxIterations);
 return 0;
} 
Output:
Enter the number of samples: 3
Enter the number of inputs: 4 Enter 
the input values for each sample:
Sample 1:
2
5
3
7
Sample 2:
2
8
6
9
Sample 3:
3
8
4
6
Enter the output values for each sample:
Sample 1: 2
Sample 2: 4
Sample 3: 8
Enter the learning rate: 5
Enter the maximum number of iterations: 3
Perceptron Learning Rule did not converge within the maximum number of iterations.







Q.7 Write a program to implement Hebb’s Rule
Program:
#include <stdio.h>
#define MAX_SIZE 100
// Function to apply Hebb's Rule
void hebbRule(double inputs[][MAX_SIZE], double weights[], int numSamples, int 
numInputs) {
 for (int i = 0; i < numSamples; i++) { 
for (int j = 0; j < numInputs; j++) {
 weights[j] += inputs[i][j] * inputs[i][j];
 }
 }
}
int main() {
 int numSamples, numInputs;
 printf("Enter the number of samples: "); 
scanf("%d", &numSamples);
 printf("Enter the number of inputs: ");
 scanf("%d", &numInputs);
 double inputs[MAX_SIZE][MAX_SIZE];
 printf("Enter the input values for each sample:\n");
 for (int i = 0; i < numSamples; i++) { 
printf("Sample %d:\n", i + 1); for 
(int j = 0; j < numInputs; j++) {
 scanf("%lf", &inputs[i][j]);
 }
 }
 double weights[MAX_SIZE] = {0.0};
 hebbRule(inputs, weights, numSamples, numInputs);
 printf("Weights: "); for (int i = 
0; i < numInputs; i++) {
 printf("%.2lf ", weights[i]);
 }
 printf("\n");
 return 0;
} 
Output:
C:\Users\rutuja mundada\Desktop>myexe1
Enter the number of samples: 4 Enter 
the number of inputs: 4 Enter the 
input values for each sample:
Sample 1: 3
4
5
2
Sample 2:
6
8
5
3
Sample 3:
1
3
4
5
Sample 4:
6
4
3
2
Weights: 82.00 105.00 75.00 42.00











Q.8 Write a program to implement Feed Forward Network.
Progrm:
#include <stdio.h>
#include <math.h>
#define MAX_SIZE 100
// Activation function (sigmoid) double 
sigmoid(double x) {
 return 1 / (1 + exp(-x));
}
// Function to calculate the dot product of two vectors double 
dotProduct(double inputs[], double weights[], int size) { 
double result = 0.0; for (int i = 0; i < size; i++) {
 result += inputs[i] * weights[i];
 }
 return result;
}
// Feedforward Network function
double feedForward(double inputs[], double hiddenWeights[][MAX_SIZE], double 
outputWeights[], double hiddenLayer[], int inputSize, int hiddenSize, double *output) {
 // Calculate hidden layer activations
 for (int i = 0; i < hiddenSize; i++) {
 double hiddenInput = dotProduct(inputs, hiddenWeights[i], inputSize); 
hiddenLayer[i] = sigmoid(hiddenInput);
 }
 // Calculate output layer activation
 double outputInput = dotProduct(hiddenLayer, outputWeights, hiddenSize); 
*output = sigmoid(outputInput);
}
int main() { int inputSize, 
hiddenSize;
 printf("Enter the number of input neurons: ");
 scanf("%d", &inputSize);
 printf("Enter the number of hidden neurons: "); 
scanf("%d", &hiddenSize);
 double inputs[MAX_SIZE];
 double hiddenWeights[MAX_SIZE][MAX_SIZE]; 
double outputWeights[MAX_SIZE]; double 
hiddenLayer[MAX_SIZE];
 double output;
 printf("Enter the input values:\n"); 
for (int i = 0; i < inputSize; i++) { 
printf("Input %d: ", i + 1);
 scanf("%lf", &inputs[i]);
 }
 printf("Enter the hidden layer weights:\n");
 for (int i = 0; i < hiddenSize; i++) { 
printf("Hidden Neuron %d:\n", i + 1); for 
(int j = 0; j < inputSize; j++) { 
printf("Weight %d: ", j + 1);
 scanf("%lf", &hiddenWeights[i][j]);
 }
 }
 printf("Enter the output layer weights:\n");
 for (int i = 0; i < hiddenSize; i++) { 
printf("Weight %d: ", i + 1);
 scanf("%lf", &outputWeights[i]);
 }
 feedForward(inputs, hiddenWeights, outputWeights, hiddenLayer, inputSize, hiddenSize, 
&output);
 printf("Output: %.4lf\n", output);
 return 0;
}
Output:
C:\Users\rutuja mundada\Desktop>myexe2
Enter the number of input neurons: 4 Enter 
the number of hidden neurons: 3 Enter the 
input values:
Input 1: 2
Input 2: 4
Input 3: 2
Input 4: 6
Enter the hidden layer weights:
Hidden Neuron 1:
Weight 1: 4
Weight 2: 2
Weight 3: 3
Weight 4: 1 Hidden 
Neuron 2:
Weight 1: 3
Weight 2: 5
Weight 3: 6
Weight 4: 2 Hidden 
Neuron 3:
Weight 1: 7
Weight 2: 5
Weight 3: 3
Weight 4: 1
Enter the output layer weights:
Weight 1: 8
Weight 2: 9
Weight 3: 6
Output: 1.0000










Q.11 Write a program to develop supervised learning algorithm
Program:
#include <stdio.h>
#include <stdlib.h>
#define MAX_SIZE 100
// Activation function (unit step) int 
unitStep(double x) {
 if (x >= 0) { 
return 1; } 
else {
 return -1;
 }
}
// Function to calculate the dot product of two vectors double 
dotProduct(double inputs[], double weights[], int size) { 
double result = 0.0; for (int i = 0; i < size; i++) {
 result += inputs[i] * weights[i];
 }
 return result;
}
// Function to train the perceptron
void train(double dataset[][MAX_SIZE], int labels[], double weights[], double learningRate, 
int dataSize, int inputSize, int maxIterations) { int iteration = 0; int error = 1;
 while (iteration < maxIterations && error != 0) { 
error = 0;
 for (int i = 0; i < dataSize; i++) {
 double prediction = unitStep(dotProduct(dataset[i], weights, inputSize)); 
int errorAmount = labels[i] - prediction; if (errorAmount != 0) {
 error = 1;
 for (int j = 0; j < inputSize; j++) {
 weights[j] += learningRate * errorAmount * dataset[i][j];
 }
 }
 }
 iteration++;
 }
}
int main() {
 int inputSize, dataSize, maxIterations;
 printf("Enter the number of input neurons: "); 
scanf("%d", &inputSize);
 printf("Enter the number of samples in the dataset: "); 
scanf("%d", &dataSize);
 double dataset[MAX_SIZE][MAX_SIZE];
 int labels[MAX_SIZE];
 printf("Enter the dataset values:\n"); 
for (int i = 0; i < dataSize; i++) { 
printf("Sample %d:\n", i + 1); for 
(int j = 0; j < inputSize; j++) { 
printf("Input %d: ", j + 1);
 scanf("%lf", &dataset[i][j]);
 }
 printf("Label: ");
 scanf("%d", &labels[i]);
 }
 // Initialize weights to zero
 double weights[MAX_SIZE] = {0};
 double learningRate; 
printf("Enter the learning rate: ");
 scanf("%lf", &learningRate);
 printf("Enter the maximum number of iterations: "); 
scanf("%d", &maxIterations);
 // Train the perceptron
 train(dataset, labels, weights, learningRate, dataSize, inputSize, maxIterations);
 // Print the weights
 printf("The weights are:\n"); 
for (int i = 0; i < inputSize; i++) {
 printf("w%d = %lf\n", i + 1, weights[i]);
 }
 // Test the perceptron 
printf("Enter a test sample:\n"); 
double testInput[MAX_SIZE];
 for (int i = 0; i < inputSize; i++) { 
printf("Input %d: ", i + 1);
 scanf("%lf", &testInput[i]);
 }
 int prediction = unitStep(dotProduct(testInput, weights, inputSize)); 
printf("Prediction: %d\n", prediction);
 return 0;
} 
Output:
C:\Users\rutuja mundada\Desktop>myexe
Enter the number of input neurons: 4 Enter 
the number of samples in the dataset: 2 Enter 
the dataset values: Sample 1:
Input 1: 3
Input 2: 4
Input 3: 2
Input 4: 6
Label: 1 Sample 
2:
Input 1: 4
Input 2: 7
Input 3: 8
Input 4: 3
Label: 2
Enter the learning rate: 3
Enter the maximum number of iterations: 3
The weights are: w1 = 36.000000 w2 = 
63.000000 w3 = 72.000000 w4 = 
27.000000
Enter a test sample:
Input 1: 2
Input 2: 4
Input 3: 6
Input 4: 8
Prediction: 1
