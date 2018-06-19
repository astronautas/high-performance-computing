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

void multiply_non_vectorizable(int n, double ** a, double ** b, double ** c);
void multiply(int n, double ** a, double ** b, double ** c);

void fillMatrix(int n, double ** matrix);

void printMatrixByRows(int n, double ** matrix);
void printMatrixByRowsInFile(int n, double ** matrix, char filename[]);

double ** createMatrix(int n);

int main(int argc, char * argv[]) {
	unsigned int mSize = 0, runs, i, option;
	struct timespec t1, t2, dt;
	double time, flops, gFlops;
	double ** a, ** b, ** c;

    if (argc == 3 && isdigit(argv[1][0]))
    {
      mSize = atoi(argv[1]);
	  option = atoi(argv[2]);
    } else if (argc == 2 && isdigit(argv[1][0])) {
	  mSize = atoi(argv[1]);
	  option = 0;
	} else {
        printf("USAGE\n   %s [SIZE]\n", argv[0]);
        return 0;
    }

	get_clockres(&t1);
	printf("Timer resolution is %lu nano seconds.\n",t1.tv_nsec);

	a = (double**)createMatrix(mSize);
	b = (double**)createMatrix(mSize);
	c = (double**)createMatrix(mSize);

	fillMatrix(mSize, a);
	fillMatrix(mSize, b);

	flops = (double)mSize * (double)mSize * (double)mSize * 2.0;

	printf("Starting benchmark with mSize = %d. \n",mSize);

	runs = time = 0;

	while (runs < 100) {

	    for (i = 0; i < mSize*mSize; i++) {
	            c[0][i] = 0;
	    }

	   	get_time(&t1);

		if (option == 1) {
	    	multiply(mSize, a, b, c);
		} else {
			multiply_non_vectorizable(mSize, a, b, c);
		}

	    get_time(&t2);

	    if ((t2.tv_nsec - t1.tv_nsec) < 0) {
	        dt.tv_sec = t2.tv_sec - t1.tv_sec - 1;
	        dt.tv_nsec = 1000000000 - t1.tv_nsec + t2.tv_nsec;
	    } else {
	        dt.tv_sec = t2.tv_sec - t1.tv_sec;
	        dt.tv_nsec = t2.tv_nsec - t1.tv_nsec;
	    }

	    time += dt.tv_sec + (double)(dt.tv_nsec)*0.000000001;
	    runs ++;
	}

	gFlops = ((flops/1073741824.0)/time)*runs;
	printf("MATRIX SIZE: %i, GFLOPS: %f, RUNS: %i\n",mSize, gFlops, runs);

	printMatrixByRows(mSize, c);

	printf ("Mean execution time: %f\n", (time/runs));

	free(a[0]);
	free(b[0]);
	free(c[0]);
}

void multiply_non_vectorizable(int n, double ** a, double ** b, double ** c) {
	int i, j, k;

	//	Naive Matrix Multiplication
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			for (k = 0; k < n; k++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}

void multiply(int n, double ** a, double ** b, double ** c) {
	int i, j, k;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			// The naive example contains all three data dependencies that prevent vectorization
			// Read after write, write after read and write after write.
			// With the provided accumulator local variable, Intel compiler is able to recognize
			// the operation as reduction, thus vectorizing it.
			// Sources: 
			// * https://ofekshilon.com/2012/05/09/a-day-with-vs11-beta-part-2-auto-vectorizer/
			// * https://software.intel.com/en-us/articles/cdiag15048
			double tmp = 0;

			for (k = 0; k < n; k++) {
				tmp += a[i][k] * b[k][j];
			}

			c[i][j] = tmp;
		}
	}
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

void fillMatrix(int n, double ** matrix) {
	int i;
	for (i = 0; i < n*n; i++) {
		matrix[0][i] = (rand()%10) - 5; //between -5 and 4
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