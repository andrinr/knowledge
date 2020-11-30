# Cuda

refer to : https://developer.nvidia.com/blog/even-easier-introduction-cuda/

or https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

## Compile & run

````bash
nvcc file.cu
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

