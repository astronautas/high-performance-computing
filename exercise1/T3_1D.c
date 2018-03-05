#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>


void get_time(struct timespec* t) {
    clock_gettime(CLOCK_MONOTONIC, t);
}

void get_clockres(struct timespec* t) {
    clock_getres(CLOCK_MONOTONIC, t);
}


void multiply(int n, double * a, double * b, double * c);


void fillMatrix(int n, double * matrix);

void printMatrixByRows(int n, double * matrix);
void printMatrixByRowsInFile(int n, double * matrix, char filename[]);

double * createMatrix(int n);



int main(int argc, char * argv[]) {
	unsigned int mSize = 0, runs, i;
	struct timespec t1, t2, dt;
	double time, flops, gFlops;
	double * a, * b, * c;

    if (argc == 2 && isdigit(argv[1][0])) {
        mSize = atoi(argv[1]);
    }else {
        printf("USAGE\n   %s [SIZE] \n", argv[0]);
        return 0;
    }

	get_clockres(&t1);
	printf("Timer resolution is %lu nano seconds.\n",t1.tv_nsec);

	a = (double*)createMatrix(mSize);
	b = (double*)createMatrix(mSize);
	c = (double*)createMatrix(mSize);

	fillMatrix(mSize, a);
	fillMatrix(mSize, b);

	flops = (double)mSize * (double)mSize * (double)mSize * 2.0;

	printf("Starting benchmark with mSize = %d.\n",mSize);

	runs = time = 0;

	while (runs < 5) {

	    for (i = 0; i < mSize*mSize; i++) {
	          // TODO initialize matrix c
	    }

	   	get_time(&t1);

	    multiply(mSize, a, b, c);

	    get_time(&t2);

	    if ((t2.tv_nsec - t1.tv_nsec) < 0) {
	        dt.tv_sec = t2.tv_sec - t1.tv_sec - 1;
	        dt.tv_nsec = 1000000000 - t1.tv_nsec + t2.tv_nsec;
	    }else {
	        dt.tv_sec = t2.tv_sec - t1.tv_sec;
	        dt.tv_nsec = t2.tv_nsec - t1.tv_nsec;
	    }

	    time += dt.tv_sec + (double)(dt.tv_nsec)*0.000000001;
	    runs ++;
	}

	gFlops = ((flops/1073741824.0)/time)*runs;
	printf("MATRIX SIZE: %i, GFLOPS: %f, RUNS: %i\n",mSize, gFlops, runs);

  /* You can use either
  printMatrixByRows(mSize, c);
  or
  printMatrixByRowsInFile(mSize, c, "asd.txt");
  to verify your implementation */

	printf ("Mean execution time: %f\n", (time/runs));

  // TODO free allocated matrices a,b, and c

}


void multiply(int n, double * a, double * b, double * c) {

	int i, j, k;
//	TODO Matrix Multiplication



}

double * createMatrix(int n) {

// TODO allocate matrix as 1 dimension array
  	return 0; // TODO return the allocated matrix 
}

void fillMatrix(int n, double * matrix) {

  // TODO fill matrix with random numbers
}


void printMatrixByRows(int n, double * matrix) {

 // TODO print matrix


}

void printMatrixByRowsInFile(int n, double * matrix, char filename[]) {

  // TODO print matrix in file

}
