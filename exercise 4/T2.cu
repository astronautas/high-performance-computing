#include <stdio.h>

__global__ void matrix_multiplication (double * d_a, double * d_b, double *d_c, int width)
{
	int k; double sum = 0;
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(col < width && row < width)
	{
		for (k = 0; k < width; k++)
			sum += d_a[row * width + k] * d_b[k * width + col];
		d_c[row * width + col] = sum;
	}

}

extern "C" int invoke_cuda_matrix_multiplication(double * h_a, double *h_b, double *h_c, int size, double ** d_a, double ** d_b, double ** d_c)
{
	// Allocate the matrix d_a, d_b, and d_c at the device memory
        int data_length = size * size * sizeof(double);

        cudaMalloc(d_a, data_length);
	cudaMalloc(d_b, data_length);
	cudaMalloc(d_c, data_length);	
        
        cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
    printf("cudaCheckError() failed 31 : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
    }

    // Copy input matrix h_a and h_b to the device memory
    cudaMemcpy(*d_a, h_a, data_length, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_b, h_b, data_length, cudaMemcpyHostToDevice);

    err = cudaGetLastError();
 	if ( cudaSuccess != err )
    {
    printf("cudaCheckError() failed 42 : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
    }

    // Launch your matrix multiplication kernel - hint do not wait for results and do not free cuda resources see the function below (get_results)
    int blocksPerMatrixRow = 2;
    int threadsPerBlocks = size / blocksPerMatrixRow; // as we choose 2 blocks per one matrix row
    matrix_multiplication<<< dim3(blocksPerMatrixRow, size), dim3(threadsPerBlocks)>>>(*d_a, *d_b, *d_c, size);

    err = cudaGetLastError();

    if ( cudaSuccess != err )
    {
    printf("cudaCheckError() failed 55: %s\n", cudaGetErrorString( err ) );
            exit( -1 );
    }

    return 0;
}

extern "C" void get_results(double * h_c, double * d_a, double * d_b, double * d_c, int data_length)
{

	cudaError err;

	cudaDeviceSynchronize();
        
	cudaMemcpy(h_c, d_c, data_length, cudaMemcpyDeviceToHost);
        
        err = cudaGetLastError();
        if ( cudaSuccess != err )
        {
                printf("cudaCheckError() failed 75: %s\n", cudaGetErrorString( err ) );
        }
        cudaFree(d_c);
        cudaFree(d_a);
        cudaFree(d_b);
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