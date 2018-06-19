#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h> /* memset */

void outMatrix(double *matrix, int n)
{
    int i,j;
     for (i=0;i<n;i++)
          {
           for (j=0;j<n;j++)
               printf("%f ",matrix[i*n+j]);
           printf("\n");
         }
}

void fillMatrix(double *matrix,int n)
{
	int i;
	for (i = 0; i < n*n; i++)
    {
    //TODO fill upper triangular of the matrix only with values, lower part with zeros
       matrix[i] = (rand()%10) - 5; //between -5 and 4
    }
}

double * createMatrix(int n) {
	
	double * m = (double*) malloc(n*n*sizeof(double));
    memset(m, 0, n*n*sizeof(double));
	
	return m;
}


int main(int argc, char* argv[])  
{    
    int size, rank;
    double t1,t2;
    double sum;
    double * a;
    double * b;
    double * c;

    int n;
    int num_threads; 
    if (argc == 2 && isdigit(argv[1][0]))
    {	
        n = atoi(argv[1]);
        // num_threads=atoi(argv[2]); 
    } else {
        printf("USAGE\n   mult [SIZE]\n");
        return 0;
    }	

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size ( MPI_COMM_WORLD ,&size );

    if (rank == 0) {
        t1=MPI_Wtime();
    }

    int block_size = n / size;

    MPI_Win a_win;
    MPI_Win b_win;
    MPI_Win c_win;

    MPI_Win_allocate(n*n*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &a, &a_win);
    MPI_Win_allocate(n*n*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &b, &b_win);
    MPI_Win_allocate(n*n*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &c, &c_win);
        
    if (rank == 0) {
        
        printf("Executing code on %d ranks\n", size);
        fillMatrix(a,n);
        fillMatrix(b,n);
    }

    // Wait for matrices to be filled
    MPI_Barrier(MPI_COMM_WORLD);

    double *local_a = createMatrix(n);
    double *local_b = createMatrix(n);
    double *local_c = createMatrix(n);

    MPI_Win_fence(0, a_win); 
    MPI_Win_fence(0, b_win); 
    MPI_Win_fence(0, c_win);

    MPI_Get(local_a + block_size*rank*n, block_size*n, MPI_DOUBLE, 0, block_size*rank*n, block_size*n, MPI_DOUBLE, a_win);

    MPI_Get(local_b, n*n, MPI_DOUBLE, 0, 0, n*n, MPI_DOUBLE, b_win);

    printf("I am rank: %d, processing: %d-%d \n----\n", rank, rank * block_size, rank * block_size + block_size);

    for(int i = rank * block_size; i < rank * block_size + block_size; i++)
    {
        for(int j=0;j<n;j++)
        {
            double sum = 0;

            for(int k=0; k < n; k++) {
                sum+=local_a[i*n+k]*local_b[k*n+j];
            }

            local_c[j+n*i] = sum;
        }
    }

    printf("I am rank: %d, putting...\n", rank);

    MPI_Put(local_c + block_size*n*rank, block_size*n, MPI_DOUBLE, 0, block_size*n*rank, block_size*n, MPI_DOUBLE, c_win);

    MPI_Win_fence(0, a_win); 
    MPI_Win_fence(0, b_win); 
    MPI_Win_fence(0, c_win);
    
    printf("I am rank: %d, finished putting\n", rank);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {

        if (n <= 5) {
            printf("\nA:\n");
            outMatrix(a, n);
            printf("B:\n");
            outMatrix(b, n);
            printf("C:\n");
            outMatrix(c, n);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_free(&a_win);
    MPI_Win_free(&b_win);
    MPI_Win_free(&c_win);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        t2 = MPI_Wtime();

        //elapsed time
        double time = t2-t1;
        printf ("Execution time: %lf\n", time);
    }

	MPI_Finalize();
    return 0;
} 

