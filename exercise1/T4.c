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

void multiply(int n, double ** a, double ** b, double ** c);

void fillMatrix(int n, double ** matrix);

void printMatrixByRows(int n, double ** matrix);
void printMatrixByRowsInFile(int n, double ** matrix, char filename[]);

double ** createMatrix(int n);

int main(int argc, char * argv[]) {
	unsigned int mSize = 0, runs, i, unrollSize;
	struct timespec t1, t2, dt;
	double time, flops, gFlops;
	double ** a, ** b, ** c;

    if (argc == 2 && isdigit(argv[1][0]))
    {
      mSize = atoi(argv[1]);
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

	printMatrixByRows(mSize, c);

	printf ("Mean execution time: %f\n", (time/runs));

	free(a[0]);
	free(b[0]);
	free(c[0]);
}

// Unrolled inner loop for 8 elements
void multiply(int n, double ** a, double ** b, double ** c) {
	int i, j, k;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {

			// Explanaton why the original code could not be optimized
			// Compiler suspects a possibility of not being able to vectorize
			// "Read after write issue", as the left side of expression could be aliased by some members of the right side
			// example scenario: a[i][k] -> c[i][j-1] (a points to c)
			// in this case, as the c[i][j] += a[i][k] and c[i][j] += a[i][k+1] executed together
			// a[i][k+1] would read incorrect c[i][j] value as it has to be computed by first iteration first
			// and then used (so the compiler suspects)
			// tmp does not point to same memory location because it is created outside the loop
			// and there are no pointer assignments
			// https://software.intel.com/sites/default/files/8c/a9/CompilerAutovectorizationGuide.pdf [4.2]
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