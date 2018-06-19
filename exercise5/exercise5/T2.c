#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>

typedef int bool;
#define true 1
#define false 0

void get_time(struct timespec* t) {
    clock_gettime(CLOCK_MONOTONIC, t);
}

void get_clockres(struct timespec* t) {
    clock_getres(CLOCK_MONOTONIC, t);
}


void multiply(int n, double ** a, double ** b, double ** c);


void fillMatrix(int n, double ** matrix);



void printMatrixByRows(int n, double ** matrix);
void printMatrixByRowsNonSquare(int rows, int cols, double ** matrix);

double ** createMatrix(int n);
double ** createMatrixNonSquare(int rows, int cols);

double ** addChecksumRowsCols(double ** matrix, int n, bool addExtraRow, bool extraColumn);
double ** calculateChecksums(double ** matrix, int n, bool hasExtraRow, bool hasExtraCol);

void injectSDC(int n, double ** matrix);
double ** detect_correct(int n, double **matrix);

int main(int argc, char * argv[]) {
	unsigned int mSize = 0, runs, i;
	struct timespec t1, t2, dt;
	double time, flops, gFlops;
	double ** a, ** b, ** c;

    if (argc == 2 && isdigit(argv[1][0])) {
        mSize = atoi(argv[1]);
    }else {
        printf("USAGE\n   %s [SIZE] \n", argv[0]);
        return 0;
    }

	get_clockres(&t1);
	printf("Timer resolution is %lu nano seconds.\n",t1.tv_nsec);

	a = (double**)createMatrix(mSize);
	b = (double**)createMatrix(mSize);
	c = (double**)createMatrix(mSize);
	
	// Add additional rows/cols for checksums
	a = addChecksumRowsCols(a, mSize, true, false);
	b = addChecksumRowsCols(b, mSize, true, false);
	c = addChecksumRowsCols(c, mSize, true, true);
	
	// Fill matrices with random numbers
	// Checksum rows do not get filled yet
    fillMatrix(mSize, a);
    fillMatrix(mSize, b);

	// Calculate checksums
	a = calculateChecksums(a, mSize, true, false);
	b = calculateChecksums(b, mSize, false, true);

	flops = (double)mSize * (double)mSize * (double)mSize * 2.0;

	printf("Starting benchmark with mSize = %d.\n",mSize);

	time = 0;

   	get_time(&t1);

	// printf("A:\n");
	// printMatrixByRowsNonSquare(mSize+1, mSize, a);

	// printf("B:\n");
	// printMatrixByRowsNonSquare(mSize, mSize+1, b);

    multiply(mSize+1, a, b, c);

	// printf("ORIGINAL\n");
	// printMatrixByRows(mSize+1, c);
	
	printf("\n----\n");
	printf("C original:\n");
	printMatrixByRows(mSize+1, c);
	printf("\n----\n");

    injectSDC(mSize,c);

	// printf("MODIFIED\n");
	// printMatrixByRows(mSize+1, c);

     get_time(&t2);

	    if ((t2.tv_nsec - t1.tv_nsec) < 0) {
	        dt.tv_sec = t2.tv_sec - t1.tv_sec - 1;
	        dt.tv_nsec = 1000000000 - t1.tv_nsec + t2.tv_nsec;
	    }else {
	        dt.tv_sec = t2.tv_sec - t1.tv_sec;
	        dt.tv_nsec = t2.tv_nsec - t1.tv_nsec;
	    }

	    time += dt.tv_sec + (double)(dt.tv_nsec)*0.000000001;
	    

	gFlops = ((flops/1073741824.0)/time);
	printf("MATRIX SIZE: %i, GFLOPS: %f\n",mSize, gFlops);

	printf("\n----\n");
	printf("C corrupted:\n");
	printMatrixByRows(mSize+1, c);
	printf("\n----\n");

  	c = detect_correct(mSize+1, c);

	printf("C corrected:\n");
	printMatrixByRows(mSize+1, c);
	printf("\n----\n");

	printf ("Mean execution time: %f\n", (time/runs));

	free(a[0]);
	free(b[0]);
	free(c[0]);
}



double ** detect_correct(int n, double **matrix)
{
	 double ** copyMatrix = (double**)createMatrix(n);

	// Make a copy of matrix without checksums
	 for (int i = 0; i < n-1; i++) {
		 for (int j = 0; j < n-1; j++) {
			copyMatrix[i][j] = matrix[i][j];
		 }
	 }

	// Compute actual checksums (n-1 as additional row/column is taken into account by the method itself)
	copyMatrix = calculateChecksums(copyMatrix, n-1, true, false);
	copyMatrix = calculateChecksums(copyMatrix, n-1, false, true);

	// Pinpoint which checksums differ. Use indexes to find the location of the error
	int errorRow;
	int errorCol;

	for (int i = 0; i < n; i++) {
		if (copyMatrix[i][n-1] != matrix[i][n-1]) {
			errorRow = i;
			break;
		}
	}

	for (int j = 0; j < n; j++) {
		if (copyMatrix[n-1][j] != matrix[n-1][j]) {
			errorCol = j;
			break;
		}
	}

	// Calculate row sums
	double errorRowSum = 0;
	for (int j = 0; j < n-1; j++) {
		if (errorCol != j) {
			errorRowSum += matrix[errorRow][j];
		}
	}

	double errorColSum = 0;
	for (int i = 0; i < n-1; i++) {
		if (errorRow != i) {
			errorColSum += matrix[i][errorCol];
		}
	}


	// Solve equations to correct value and replace the incorrect value
	int trueValue1 = matrix[errorRow][n-1] - errorRowSum;
	int trueValue2 = matrix[n-1][errorCol] - errorColSum;

	if (trueValue1 != trueValue2) {
		printf("There has been one than more error or the checksums have been corrupted. Exiting...");
		exit(0);
	}

	matrix[errorRow][errorCol] = trueValue1;

	free(copyMatrix[0]);
	
	return matrix;
}


void multiply(int n, double ** a, double ** b, double ** c) {
	int i, j, k;
        /*Naive Matrix Multiplication*/
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			for (k = 0; k < n-1; k++) {
                //printf("i: %d, j, %d, k: %d, c: %lf \n", i, j, k, c[i][j]);
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}

}


void injectSDC(int n, double ** matrix)
{
  //TODO ....randomly change the value of an element in the matrix
  int row = (rand() % n);
  int col = (rand() % n);
  int val = (rand() % 10) - 5;

  matrix[row][col] = val;
}

double ** createMatrix(int n) {
	int i;
	double ** matrix = (double**) calloc(n,sizeof(double*));
	double * m = (double*) calloc(n*n,sizeof(double));
	for (i = 0; i < n; i++) {
		matrix[i] = m+(i*n);
	}
	return matrix;
}

double ** createMatrixNonSquare(int rows, int cols) {
	int i;
	double ** matrix = (double**) calloc(rows,sizeof(double*));
	double * m = (double*) calloc(rows*cols,sizeof(double));

	for (i = 0; i < rows; i++) {
		matrix[i] = m+(i*cols);
	}

	return matrix;
}

double ** addChecksumRowsCols(double ** matrix, int n, bool addExtraRow, bool addExtraCol) {
	double ** newMatrix;

	if (addExtraRow == true && addExtraCol == false) {
		// Matrix a
		newMatrix = createMatrixNonSquare(n+1, n);
	} else if (addExtraRow == false && addExtraCol == true) {
		// Matrix b
		newMatrix = createMatrixNonSquare(n, n+1);
	} else {
		// Matrix c
		newMatrix = createMatrixNonSquare(n+1, n+1);
	}

	free(matrix[0]);
	return newMatrix;
}

double ** calculateChecksums(double ** matrix, int n, bool hasExtraRow, bool hasExtraCol) {
	if (hasExtraRow == true && hasExtraCol == false) {
		// Calculate checksums
		// Iterate over columns
		for (int j = 0; j < n; j++) {
			double sum = 0;
			
			// Sum up whole current column (i.e. iterate over rows for current column)
			for (int i = 0; i < n; i++) {
				sum += matrix[i][j];
			}

			matrix[n][j] = sum;
		}
	} else if (hasExtraRow == false && hasExtraCol == true) {
		// Calculate checksums
		// Iterate over rows
		for (int i = 0; i < n; i++) {
			double sum = 0;

			// Sum up whole current row (i.e. iterate over columns for current row)
			for (int j = 0; j < n; j++) {
				sum += matrix[i][j];
			}

			matrix[i][n] = sum;
		}
	}

	return matrix;
}


void fillMatrix(int n, double ** matrix) {
	
        int i, j; 

        for (i = 0; i < n; i++)
        {
           for(j = 0; j< n ; j++)
           {
                matrix[i][j] = (rand()%10) - 5; //between -5 and 4
           }
        }
}

void printMatrixByRows(int n, double ** matrix) {
	int i, j;

	printf("{");
	for (i = 0; i < n; i++) {
		printf("[");
		for (j = 0; j < n; j++) {
			printf("%d",(int)matrix[i][j]);
			if (j != n-1)
				printf(",");
			else
				printf("]");
		}
		if (i != n-1)
			printf(",\n");
	}
	printf("}\n");
}

void printMatrixByRowsNonSquare(int rows, int cols, double ** matrix) {
	int i, j;

	printf("\n-----\n");
	printf("{");
	for (i = 0; i < rows; i++) {
		printf("[");
		for (j = 0; j < cols; j++) {
			printf("%d",(int)matrix[i][j]);
			if (j != cols-1)
				printf(",");
			else
				printf("]");
		}
		if (i != rows-1)
			printf(",\n");
	}
	printf("}");
	printf("\n-----\n");
}

void printMatrixByRowsInFile(int n, double **matrix, char filename[]) {
	int i, j;

	FILE *fp = fopen(filename, "w");

	fprintf(fp, "{");
	for (i = 0; i < n; i++) {
		fprintf(fp, "[");
		for (j = 0; j < n; j++) {
			fprintf(fp, "%d",(int)matrix[i][j]);
			if (j != n-1)
				fprintf(fp, ",");
			else
				fprintf(fp, "]");
		}
		if (i != n-1)
			fprintf(fp, ",\n");
	}
	fprintf(fp, "}\n");
	fclose(fp);
}
