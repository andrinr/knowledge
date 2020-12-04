# C++ / C

Header files are not required but recommended.

## Compiling 

````bash
# Compilation, display warnings, optimize and add additional info
g++ -Wall -O -g file.cpp 
````

## Running

````bash
# Simply run the produced .out file, usually a.out
./a.out
````



## Pointers

````c++
a = 25;
b = a; //store the value, b = 25
c = &a; // store the adress of a, c is now a pointer, c=adress
d = *c; // deference operator, now d = 25
````

``&`` is the “address of” operator

``*`` is the “value of” operator

````c++
// Declaring a pointer
type * name;
````


