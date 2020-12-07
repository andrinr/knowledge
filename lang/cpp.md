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

### Example

````c++
#include <iostream>
int main() {
  int * pointer;
  std::cout << pointer << std::endl;
  std::cout << &pointer << std::endl;
  
  int value = 10;
  std::cout << value << std::endl;
  std::cout << &value << std::endl;
  
  pointer = &value;
  std::cout << *pointer << std::endl;
  std::cout << &pointer << std::endl;
  return 0;
}
````

This program will output:

````
0
0x7ffff654d2d8
10
0x7ffff654d2d4
10
0x7ffff654d2d8
````