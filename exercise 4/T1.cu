#define N 1024 // number of rows = number of columns
#include <time.h>
#include <stdio.h>

__global__ void matrix_mult_kernel (int *a, int *b, int *c, int n);
void init(int * input,int length);
void print_matrix(int * matrix, int size);

int main()
{
	clock_t start, end;
	double cpu_time_used;

	int * h_a,*d_a;
	int * h_b,*d_b;
	int * h_c,*d_c;
	int data_length = N * N * sizeof(int);

	h_a=(int*)malloc(data_length);
	h_b=(int*)malloc(data_length);
	h_c=(int*)malloc(data_length);

	init(h_a,N*N);
	init(h_b,N*N);

	// Initialize matrices on the gpu
	cudaMalloc(&d_a, data_length);
	cudaMalloc(&d_b, data_length);
	cudaMalloc(&d_c, data_length);	

	cudaError err = cudaGetLastError();

	if ( cudaSuccess != err )
	{
		printf("cudaCheckError() failed line 29 : %s\n", cudaGetErrorString( err ) );
		exit( -1 );
	}

	start = clock();

	// Copy matrices to the gpu
	cudaMemcpy(d_a, h_a, data_length, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, data_length, cudaMemcpyHostToDevice);
	
	err = cudaGetLastError();
	
	if ( cudaSuccess != err )
	{
		printf("cudaCheckError() failed line 42: %s\n", cudaGetErrorString( err ) );
		exit( -1 );
	}
	 
	int blocksPerMatrixRow = 2;
	int threadsPerBlocks = N / blocksPerMatrixRow; // as we choose 2 blocks per one matrix row
	matrix_mult_kernel<<< dim3(blocksPerMatrixRow, N), dim3(threadsPerBlocks)>>>(d_a, d_b, d_c, N);

	err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
	printf("cudaCheckError() failed line 51 %s\n", cudaGetErrorString( err ) );
		exit( -1 );
	}

	// Copy output matrix h_c to the host memory
	cudaMemcpy(h_c, d_c, data_length, cudaMemcpyDeviceToHost);

 	err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
	printf("cudaCheckError() failed line 61: %s\n", cudaGetErrorString( err ) );
		exit( -1 );
	}

	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	printf("Elapsed time: %f\n", cpu_time_used);

	// Free all alocated memory
	err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		printf("cudaCheckError() failed line 70: %s\n", cudaGetErrorString( err ) );
		exit( -1 );
	}

	free(h_a);
	free(h_b);
 	free(h_c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

void init(int * input, int size)
{
	int i;
	for(i=0;i<size;i++)
	{
		input[i]=rand()%5;
	}
}

void print_matrix(int * matrix,int size)
{
	printf("Matrix items: \n");
	int i,j;
	for(i=0;i<size;i++)
	{
		for(j=0;j<size;j++)
			printf("%d,",matrix[i*size+j]);
		printf("\n");
	}
}

__global__ void matrix_mult_kernel (int *a, int *b, int *c, int n)
 {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	
	int i, j, k;
	
	// Each thread calculate one cell of c
	for (i = y; i < y + 1; i++) {
		for (j = x; j < x + 1; j++) {
			for (k = 0; k < n; k++) {
				c[i*n+j] += a[i*n+k] * b[k*n+j];
			}
		}
	}
 }
