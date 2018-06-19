#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#define MIN(a,b) ((a) < (b) ? a : b)

void get_time(struct timespec* t) {
    clock_gettime(CLOCK_MONOTONIC, t);
}

void get_clockres(struct timespec* t) {
    clock_getres(CLOCK_MONOTONIC, t);
}


void multiply(int n, double * a, double * b, double * c);


void blockMultiply(int n, int bn, double * a, double * b, double * c);

void fillMatrix(int n, double * matrix);

void printMatrixByRows(int n, double * matrix);
void printMatrixByRowsInFile(int n, double * matrix, char filename[]);

double * createMatrix(int n);

int main(int argc, char * argv[]) {
	omp_set_dynamic(0);     // Explicitly disable dynamic teams
	omp_set_num_threads(4); // Use 4 threads for all consecutive parallel region

	#pragma omp parallel for schedule(static)
	for (int i = 0; i <= 100; ++i)
	{
		// Do something here
		printf_s("test() iteration UNORDERED %d\n", i);
		
		#pragma omp ordered
		printf_s("test() iteration %d\n", i);
	}
}


void multiply(int n, double * a, double * b, double * c) {
	#pragma omp parallel
	{
		 /* Obtain thread number */
		int tid = omp_get_thread_num();

		/* Only master thread does this */
		if (tid == 0) 
		{
			int nthreads = omp_get_num_threads();
			printf("Number of threads = %d\n", nthreads);
		}

		#pragma omp for
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				for (int k = 0; k < n; k++) {
					c[i*n+j] += a[i*n+k] * b[k*n+j];
				}
			}
		}
	}
}

void blockMultiply(int n, int bn, double * a, double * b, double * c) {
	#pragma omp parallel 
	{

		 /* Obtain thread number */
		int tid = omp_get_thread_num();

		/* Only master thread does this */
		if (tid == 0) 
		{
			int nthreads = omp_get_num_threads();
			printf("Number of threads = %d\n", nthreads);
		}

		#pragma omp for
		for (int ii = 0; ii < n; ii +=bn) {
			for (int jj = 0; jj < n; jj += bn) {
				for (int i = ii; i < MIN(ii+bn,n); i++) {
					for (int j = jj; j < MIN(jj+bn,n); j++) {
						for (int k = 0; k < n; k++) {
							c[i*n+j] += a[i*n+k] * b[k*n+j];
						}
					}
				}
			}
		}
	}
}

double * createMatrix(int n) {
	int i;
	double * matrix = (double*) calloc(n*n,sizeof(double));
	return matrix;
}

void fillMatrix(int n, double * matrix) {
	int i;
	for (i = 0; i < n*n; i++) {
		matrix[i] = (rand()%10) - 5; //between -5 and 4
	}
}


void printMatrixByRows(int n, double * matrix) {
	int i, j;

	printf("{");
	for (i = 0; i < n; i++) {
		printf("[");
		for (j = 0; j < n; j++) {
			printf("%d",(int)matrix[i*n+j]);
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

void printMatrixByRowsInFile(int n, double *matrix, char filename[]) {
	int i, j;

	FILE *fp = fopen(filename, "w");

	fprintf(fp, "{");
	for (i = 0; i < n; i++) {
		fprintf(fp, "[");
		for (j = 0; j < n; j++) {
			fprintf(fp, "%d",(int)matrix[i*n+j]);
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
