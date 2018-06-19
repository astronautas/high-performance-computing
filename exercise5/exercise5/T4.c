#include <mpi.h>
#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

void printMatrixByRows(int n, double * matrix);

double * createMatrix(int n);
void fillMatrix(int n, double * matrix);



int main(int argc, char* argv[]) {
	int n;
	double t1,t2;
    	double time, flops, gFlops;
    
    	if (argc == 2 && isdigit(argv[1][0])) {
        	n = atoi(argv[1]);
    	}else {
        	printf("USAGE\n   %s [SIZE]\n", argv[0]);
        	return 0;
    	}

    	int i,j,k;       

    	int size, rank;
    	double sum;
    	flops = n*n*n* 2.0;

	int nn, n_up, sizeSent, sizeToSent;
	
   	
    	MPI_Init (& argc ,& argv );
    	MPI_Comm_size ( MPI_COMM_WORLD ,& size );
   	MPI_Comm_rank ( MPI_COMM_WORLD ,& rank );
    		
    	double * a;
    	double * b;
    	double * c;
    
    	a = (double*)createMatrix(n);
    	b = (double*)createMatrix(n);
   	c = (double*)createMatrix(n);
   
    
   	fillMatrix(n, a);
  	fillMatrix(n, b);
    
    	//get time to start benchmark
    	t1=MPI_Wtime();
    

	/* Boradcast n to all processes */
    	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	nn = (n/size) + (n%size > rank);

	n_up = n * nn;

	/* Send a by splitting it in row-wise parts */
	if(rank==0) {
		sizeSent = n_up;
		for (i=1;i<size;i++) {
			sizeToSent = n * ((n/size) + (n%size > i));
			MPI_Send(a+sizeSent,sizeToSent,MPI_DOUBLE,i,1,MPI_COMM_WORLD);
			sizeSent += sizeToSent;
		}
	} else {
		MPI_Recv(a,n_up,MPI_DOUBLE,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	}

	/* Send b completely to each process */
	MPI_Bcast(b, n*n, MPI_DOUBLE,0,MPI_COMM_WORLD);

	#pragma omp parallel for private(i,j,k,sum) shared(a,b,c)
	for(i=0;i<nn;i++) {
      		for(j=0;j<n;j++) {
        		sum=0;
        		for(k=0;k<n;k++) {
          			sum+=a[i*n+k]*b[k*n+j];
			}
        		c[j+n*i] = sum;
      		}
    	}

	/* Receive partial results from each slave */
	if(rank==0) {
		sizeSent= n_up;
		for(i=1;i<size;i++) {
			sizeToSent = n*((n/size) + (n%size > i));
			MPI_Recv(c+sizeSent,sizeToSent,MPI_DOUBLE,i,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			sizeSent += sizeToSent;
		}
	} else {
		MPI_Send(c,n_up,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
	}
    
    
    	//get time to finish the measurement
    	t2 = MPI_Wtime();
    
    	//elapsed time
    	time = t2-t1;
      
    	if (rank == 0) {
        	gFlops = ((flops/1073741824.0)/time);
		
		//printMatrixByRows(n, a);
		//printMatrixByRows(n, b);
		//printMatrixByRows(n, c);

        	printf("MATRIX SIZE: %i, GFLOPS: %f\n",n, gFlops);
        	printf ("Execution time: %f\n", time);
        
    	}

	free(a);
	free(b);
	free(c);

    	MPI_Finalize ();
    	return 0;
}


double * createMatrix(int n) {
    	double * m = (double*) malloc(n*n*sizeof(double));
    
    	return m;
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
            		if (j != n-1) {
                		printf(",");
           		} else {
                		printf("]");
			}
        	}
        	if (i != n-1){
            		printf(",\n");
		}
    	}
    	printf("}\n");
}

