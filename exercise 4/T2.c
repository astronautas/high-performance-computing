#define N 1024
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

void init(double * input,int length);
void print_matrix(double * matrix, int size);
int invoke_cuda_matrix_multiplication(double * h_a, double * h_b, double * h_c, int size, double ** d_a, double ** d_b, double ** d_c);
void get_results(double *h_c, double * d_a, double * d_b, double * d_c, int data_length);
void addMatricesMpiParallel(double * a, double * b, double * output, int rowLength);

double t1,t2;
double time;

int main()
{
	MPI_Init(NULL,NULL);
	int id;
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD,&world_size);
	MPI_Comm_rank(MPI_COMM_WORLD,&id);
	int data_length = N * N * sizeof(double);

	double * h_a, * d_a;
	double * h_b, * d_b;
	double * h_c, * d_c; // will store A*B
	double * h_c2; // will store A+B
	double * h_c2_local;

	h_a=(double*)malloc(data_length);
	h_b=(double*)malloc(data_length);
	h_c=(double*)malloc(data_length);
	h_c2=(double*)malloc(data_length);
	h_c2_local = (double*)malloc(data_length);

	if (id==0)
	{
		// initialize matrix a and b;		
		omp_set_dynamic(0);     // Explicitly disable dynamic teams
		omp_set_num_threads(2); // Use 2 threads for all consecutive parallel regions

		init(h_a, N*N);
		init(h_b, N*N);

        // check time t1
		t1 = MPI_Wtime();

		// call invoke_cuda_matrix_multiplication to do A*B
		invoke_cuda_matrix_multiplication(h_a, h_b, h_c, N, &d_a, &d_b, &d_c);
	}

	// divide matrix a and matrix b between MPI ranks
	MPI_Bcast(h_a, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(h_b, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// each rank should add its part of A and B
	addMatricesMpiParallel(h_a, h_b, h_c2_local, N);

	//collect results of A+B from all other ranks
	int block_size = N / world_size;
    MPI_Gather(h_c2_local + block_size*N*id, block_size*N, MPI_DOUBLE, h_c2, block_size*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if(id==0)
	{
		// collect results of A*B from GPU
		get_results(h_c, d_a, d_b, d_c, data_length);

		//get time to finish the measurement
		t2 = MPI_Wtime();
		
		//elapsed time
		time = t2-t1;

		printf("Elapsed time: %f\n", time);

		// if N < 10 print results of addition h_c2 then print results of multiplication h_c
		if (N < 10) {
			print_matrix(h_c2, N);
		}
	}

	// free the memory you allocated
	free(h_a);
	free(h_b);
	free(h_c2);
	free(h_c);

	MPI_Finalize();
}
void init(double * input, int size)
{
	int i;
	for(i=0;i<size;i++)
	{
		input[i]=rand()%5;
	}
}

void print_matrix(double * matrix,int size)
{
	printf("Matrix items: \n");
	int i,j;
	for(i=0;i<size;i++)
	{
		for(j=0;j<size;j++)
			printf("%f,",matrix[i*size+j]);
		printf("\n");
	}
}

void addMatricesMpiParallel(double * a, double * b, double * output, int rowLength) {

	#pragma omp parallel for collapse(2)
	for (int i = 0; i < rowLength; i++) 
	{
		for (int j = 0; j < rowLength; j++) {
			output[i*rowLength + j] = a[i*rowLength + j] + b[i*rowLength + j];
		}
	}
}