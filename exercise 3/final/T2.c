#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <ctype.h>
#include <math.h>
#include "mpi.h"
#include <stdbool.h>
#include <string.h> /* memset */

#define MASTER 0


void fillMatrix(int n, double * matrix);

void printMatrixByRows(int n, double * matrix);
void printMatrixByRowsInFile(int n, double * matrix, char filename[]);
void mat_mul_chunk(int n, int start_row, int start_col, int chunk_size, double *a, double *b, double *c);
double * createMatrix(int n);



int main(int argc, char * argv[]) {
   
    int i,j;
    int METHOD = 1;
    int n = 1;
    int rank,size;
    int num_tasks;
    int start_id;
    int chunk_size;
    double *A,*B,*C,*C_out;
    double t1,t2;
    double time, flops, gFlops;
    double * final_matrix;
    int block_size;
    
    if (argc == 3 && isdigit(argv[1][0]) && isdigit(argv[2][0])) {
        n = atoi(argv[1]);
        METHOD = atoi(argv[2]);
    } else if (argc == 4) {
        n = atoi(argv[1]);
        METHOD = atoi(argv[2]);
        block_size = atoi(argv[3]);
    } else {
        printf("USAGE\n   mult [SIZE] [Scheduling method]\n");
        printf("EXAMPLE\n   %s 4000  1\n",argv[0]);
        return 0;
    }

    //TODO allocate matrices
    A = createMatrix(n);
    B = createMatrix(n);
    C = createMatrix(n);

    flops = (double) n*n*n* 2.0;
    num_tasks =  n*n;

    
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size ( MPI_COMM_WORLD ,&size );

    // if master
    if (rank == MASTER)
    {
        final_matrix = createMatrix(n);

        fillMatrix(n, A);
        fillMatrix(n, B);
    }

    //get time to start benchmark
    t1=MPI_Wtime();
    
    //broadcast the matrices to all ranks
    MPI_Bcast(&A[0],n*n,MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&B[0],n*n,MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    MPI_Request request =0;
    MPI_Status   status;

    if (METHOD == 1) //Block cyclic, block size = n/size
    {
        block_size = n / size;
        
        // Each process calculates some portion of C (line by line)
        for(int i = rank * block_size; i < rank * block_size + block_size; i++)
        {
            for(int j=0;j<n;j++)
            {
                double sum = 0;

                for(int k=0; k < n; k++) {
                    sum+=A[i*n+k]*B[k*n+j];
                }

                C[j+n*i] = sum;
            }
        }

        if (rank == 0) {
            MPI_Gather(C + block_size*n*rank, block_size*n, MPI_DOUBLE, final_matrix, block_size*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gather(C + block_size*n*rank, block_size*n, MPI_DOUBLE, NULL, block_size*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }
    else if(METHOD == 2) // finer block sizes
    {
        for (int block_start = rank*block_size; block_start < n; block_start += block_size*size) {
            
            // Iterate single block
            for(int i = block_start; i < block_start + block_size; i++)
            {
                for(int j=0;j<n;j++)
                {
                    double sum = 0;

                    for(int k=0; k < n; k++) {
                        sum+=A[i*n+k]*B[k*n+j];
                    }

                    C[j+n*i] = sum;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Receiving blocks...\n");

            for (int slave_rank = 1; slave_rank < size; slave_rank++) {
                for (int block_tag = slave_rank; block_tag < n/block_size; block_tag += size) {
                    // printf("Receiving %d block from %d rank...\n", block_tag, slave_rank);
                    MPI_Recv(C + block_tag*block_size*n, block_size*n, MPI_DOUBLE, slave_rank, block_tag, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    printf("Received %d block from %d rank\n", block_tag, slave_rank);
                }
            }
        } else {
            // Send process blocks
            for (int block_tag = rank; block_tag < n/block_size; block_tag += size) {
                printf("Sending %d block from %d rank\n", block_tag, rank);
                MPI_Send(C + block_tag*block_size*n, block_size*n, MPI_DOUBLE, 0, block_tag, MPI_COMM_WORLD);
            }
        }
    }
  
    //get time to finish the measurement
    t2 = MPI_Wtime();
    
    //elapsed time
    time = t2-t1;

    if (rank == MASTER)
    {
       
        if (METHOD == 1) {
            printf("Block cyclic scheduling (uniform)\n");

            if (n <= 5) {
                printMatrixByRows(n, final_matrix);
            }        }
        else if(METHOD == 2)
        {
            printf("\nBlock cyclic scheduling (finer block size) \n");

            if (n <= 5) {
                printMatrixByRows(n, C);
            }
        }
        

        gFlops = ((flops/1073741824.0)/time);
        printf("MATRIX SIZE: %i, GFLOPS: %lf\n",n, gFlops);
        printf ("Execution time: %lf\n", time);
        
    }

    free(A);
    free(B);
    free(C);

    if (rank == 0) {
      free(final_matrix);  
    }
    
    MPI_Finalize();
    return 0;
    
}

void mat_mul_chunk(int n, int start_row, int start_col, int chunk_size, double *a, double *b, double *c)
{
    int real_start_row = start_row * chunk_size;
    int real_start_col = start_col * chunk_size;
    
    for(int i = real_start_row; i < real_start_row + chunk_size; i++)
    {
        for(int j = real_start_col; j < real_start_col + chunk_size; j++)
        {
            double sum = 0;

            for(int k = 0; k < n; k++) {
                sum += a[i*n + k]*b[k*n+j];
            }

            c[j+n*i] = sum;
        }
    }
    
}


double * createMatrix(int n) {
	
	double * m = (double*) malloc(n*n*sizeof(double));
    memset(m, 0, n*n*sizeof(double));
	
	return m;
}

void fillMatrix(int n, double * matrix) {
    int i;
	int line = 0;

    for (i = 0; i < n*n; i += n)
    {
        // Fills line with double in triangular fashion
		for (int j = 0; j < n - line; j++) {
			matrix[i + j] = (rand()%10) - 5;
		}

        // Fills remaining cells with 0
		for (int j = n - line; j < n; j++) {
			matrix[i + j] = 0;
		}

		line += 1;
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