#include <mpi.h>
#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

void printMatrixByRows(int n, double * matrix);

double * createMatrix(int n);
void fillMatrix(int n, double * matrix);

// Algorithm:
// Each process calculates a final portion of matrix c
// Each thread gets a and b to work with
int main(int argc, char* argv[])  {    
    
    int n;
    double t1,t2;
    double time, flops, gFlops;
    
    if (argc == 2 && isdigit(argv[1][0])) {
        n = atoi(argv[1]);
    }else {
        printf("USAGE\n   %s [SIZE]\n", argv[0]);
        return 0;
    }

    int size, rank;
    flops = (double) n*n*n* 2.0;
   
    MPI_Init (& argc ,& argv );
    MPI_Comm_size ( MPI_COMM_WORLD ,&size );
    MPI_Comm_rank ( MPI_COMM_WORLD ,&rank );

    if (n % size != 0) {
        printf("Matrix size should be divisable by the MPI processes count.");
        return 0;
    }
    
    double * a;
    double * b;
    double * c;
    double * final_matrix;
    
    //get time to start benchmark
    t1 = MPI_Wtime();

    a = createMatrix(n);
    b = createMatrix(n);
    c = createMatrix(n);

    if (rank == 0) {
        final_matrix = createMatrix(n);

        fillMatrix(n, a);
        fillMatrix(n, b);
    }


    // Root sends inputs to all slave processes
    // All non-root processes fill their buffers (a,b) ~ receive
    MPI_Bcast(a, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int block_size = n / size;

    printf("I am rank: %d, processing: %d-%d \n----\n", rank, rank * block_size, rank * block_size + block_size);

    // Each process calculates some portion of C (line by line)
    #pragma omp parallel for
    for(int i = rank * block_size; i < rank * block_size + block_size; i++)
    {
      for(int j=0;j<n;j++)
      {
        double sum = 0;

        for(int k=0; k < n; k++) {
            sum+=a[i*n+k]*b[k*n+j];
        }

        c[j+n*i] = sum;
      }
    }

    if (rank == 0) {
        MPI_Gather(c + block_size*n*rank, block_size*n, MPI_DOUBLE, final_matrix, block_size*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gather(c + block_size*n*rank, block_size*n, MPI_DOUBLE, NULL, block_size*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    //get time to finish the measurement
    t2 = MPI_Wtime();
    
    //elapsed time
    time = t2-t1;
    
    if (rank == 0)
    {
        gFlops = ((flops/1073741824.0)/time);
        printf("MATRIX SIZE: %i, GFLOPS: %f:\n",n, gFlops);
        printf ("Execution time: %f\n", time);
    }

    if (rank == 0) {
        //printMatrixByRows(n, final_matrix);
    }

    free(a);
    free(b);
    free(c);

    if (rank == 0) {
      free(final_matrix);  
    }

    MPI_Finalize();

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

