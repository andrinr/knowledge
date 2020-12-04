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

# Or allocate memory only on device memory, then data needs to be copied by hand?
cudaMallocManaged(&data, N*sizeof(double));

# After finishing
cudaFree(data);
````

