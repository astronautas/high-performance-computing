#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#define MIN(a,b) ((a) < (b) ? a : b)

/* OS X */
#ifdef __MACH__
	#include <mach/mach_time.h>

	static const int GIGA = 1000000000;
	static mach_timebase_info_data_t sTimebaseInfo;

	void get_clockres(struct timespec* t) {
		if (sTimebaseInfo.denom == 0) {
			mach_timebase_info(&sTimebaseInfo);
		}
		t->tv_nsec = (uint64_t) (sTimebaseInfo.numer / sTimebaseInfo.denom);
	}

	void get_time(struct timespec* t) {
		uint64_t time = mach_absolute_time();
		time *= (sTimebaseInfo.numer / sTimebaseInfo.denom);
		t->tv_sec = time / GIGA;
		t->tv_nsec = time % GIGA;
	}

/* linux */
#elif __gnu_linux__
	void get_time(struct timespec* t) {
		clock_gettime(CLOCK_MONOTONIC, t);
	}
	void get_clockres(struct timespec* t) {
		clock_getres(CLOCK_MONOTONIC, t);
	}
#endif

void multiply(int n, double * a, double * b, double * c);

void fillMatrix(int n, double * matrix);
void fillMatrix_upper(int n, double * matrix);


double * createMatrix(int n);


int main(int argc, char * argv[]) {
	unsigned int mSize = 0, bSize = 0, runs, i;
	struct timespec t1, t2, dt;
	double time, flops, gFlops;
	double * a, * b, * c;

    if (argc == 2 && isdigit(argv[1][0]) ) {
        mSize = atoi(argv[1]);
    }else {
        printf("USAGE\n   mult [SIZE]\n");
        return 0;
    }

	get_clockres(&t1);
	printf("Timer resolution is %lu nano seconds.\n",t1.tv_nsec);

	a = (double*)createMatrix(mSize);
	b = (double*)createMatrix(mSize);
	c = (double*)createMatrix(mSize);

	fillMatrix_upper(mSize, a);
	fillMatrix(mSize, b);

	flops = (double)mSize * (double)mSize * (double)mSize * 2.0;

	printf("Starting benchmark with mSize = %i.\n",mSize);

	 time = 0;

	    for (i = 0; i < mSize*mSize; i++) {
	            c[i] = 0;
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
	 
	

	gFlops = ((flops/1073741824.0)/time);
	printf("MATRIX SIZE: %i, GFLOPS: %f\n",mSize, gFlops);

	//printMatrixByRowsInFile(mSize, c, "asd.txt");
        printf ("Mean execution time: %f\n", (time));

	free(a);
	free(b);
	free(c);
}

void multiply(int n, double * a, double * b, double * c) {
	
	int i, j, k;

	//	Optimized Naive Matrix Multiplication
	double sum;
	
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

		#pragma for shared(a,b,c) private(i,j,k, sum) collapse(2) schedule(guided)
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				sum = 0.0;
				for (k = 0; k < n; k++) {
					sum += a[i*n+k] * b[k*n+j];
				}
				c[i*n+j] = sum;
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

void fillMatrix_upper(int n, double * matrix) {
        int i;
	 
        for (i = 0; i < n*n; i++) {
		if(i/n <= i%n)
		{
			 matrix[i] = 0.0;
		}
		else 
		{
	               matrix[i] = (rand()%10) - 5; //between -5 and 4

		}
        }
}

