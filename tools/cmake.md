# CMake

CMake is a which generates makefiles, this can be used with various programming languages such as fortran, c, c++.

1. touch CMakelists.txt
2. mkdir build & cd build
3. cmake ..
4. cmake --build .



## Step by step

We set the project name with:

```cmake
project(<projectName> VERSION 1.0 CXX C)
```

When using cuda this can now be set in the project command as follows:

```cmake
project(<projectName> VERSION 1.0 CXX CUDA )
```

To add an external directory we first tell cmake to find it. Cmake will search a lot of places for the library. The locations can be logged using ``cmake --debug-path``. 

```cmake
find_package(blitz REQUIRED)
```

In case we notice that cmake is unable to find the library, we may provide additional hints: 

```cmake
set(blitz_DIR /usr/local/lib/cmake)
```

We need to provide it the location of the <libname>Config.blitz file.  



## Example MPI & CUDA cmake

```cmake
cmake_minimum_required(VERSION 3.17.0)
cmake_policy(VERSION 3.6...3.17.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(glb VERSION 0.1 LANGUAGES CXX CUDA )

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_C_FLAGS -fopenmp)
add_executable(glb src/glb.cpp src/IO.cpp src/orb.cpp src/comm/MPIMessaging.cpp src/services/services.cpp)
set_target_properties(glb PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(glb ${MPI_LIBRARIES})

set(blitz_DIR /usr/local/lib/cmake)
find_package(blitz REQUIRED)

```





