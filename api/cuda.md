# CUDA

**C**ompute **U**nified **D**evice **A**rchitecture

refer to : https://developer.nvidia.com/blog/even-easier-introduction-cuda/

or https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

## Compile & run

````bash
nvcc file.cu
./a.out
````

## Declare a function

````c++
__global__ void function(){
	
}
````

## Get stride and id

````c++
int index = threadIdx.x;
int stride = blockDim.x;
````

## CPU / GPU data transfer

````c++
# Declare a pointer
int *data;
# Hand adress of pointer to function, pass data size
# Allocated unified memory, accesible from CPU and GPU
cudaMallocManaged(&data, N*sizeof(double));

# Or handle data transfer manually which in most cases is adwised
# host memory allocation
int * h_data = (int*)malloc(N*sizeof(int));
# Declare device data pointer
int * d_data;
# Use double pointer (!) to allocate device memory
cudaMalloc(&d_data, N*sizeof(int))
# <Initialize data on host here>
# Copy data from host to device
cudaMemcpy(d_data, h_data, N*sizeof(int), cudaMemcpyHostToDevice);
# <Do calculations here>
# Copy memory back from device to host
cudaMemcpy(h_data, d_data, N*sizeof(int), cudaMemcpyDeviceToHost);

# After finishing
cudaFree(data);
````

## Shared memory, global memory and unified memory

````c++

````





## Atomic operators

Can be used everywhere, where the memory is accessible. Guarantee race condition arithmetic operations.

````c++
atomicAdd(float * adress, int value);
````

