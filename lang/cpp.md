# C++ / C

Header files are not required but recommended.

## Compile & Run

````bash
# Compilation, display warnings, optimize and add additional info
g++ -Wall -O -g file.cpp 
````

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

### Pointers and Arrays

````c++
#include <iostream>
void func(int * value, int * arr){
  std::cout << value << std::endl;
  std::cout << *value << std::endl;
  
  *value += 10;
  
  std::cout << *value << std::endl;
  std::cout << arr[0] << std::endl;
}

int main() {
  int value = 10; 
  int arr[2] = {2,4};
  
  func(&value, arr);
  return 0;
}

````

This program will output:

````c++
0x7ffd36763bac
10
20
2
````

### Vectors

````
std::vector<type> name;

````



### Double pointers

````c++
int *pointer;
int *doublePointer = &pointer;
````

TODO: Why on earth do we need double pointers?? 

## Memory allocation

````c++
int var;
int arr[100];

````



## Bit operations

````cpp
// Shift
int a = 16;
int b = a << 1;
// b = 8
````



## L and R values

https://www.internalpointers.com/post/understanding-meaning-lvalues-and-rvalues-c

```c++
int x = 666; 
// 666 is an r value. Has no specific memory address, stored on temporary register. 
// x on the other hand is an l value, it does have a specific memory location. 
```



## Abstract classes

We can define an abstract class, usually in the header file as follows:

```c++
class Bar {
    virtual void foo() = 0;
}
```

This function needs to be implemented in another header file which derives Bar also there needs to be a function definition. 



## Void pointers

Void pointers are essentially memory adresses where the datatype is not known. We can cast from a void pointer to a type as follows:

```c++
void foo(void * rawData) {
    T data = *(struct T*)rawData;
}
```

If we want to get a void pointer from typed data we can do the following:

```c++
void * rawData = &output;
```

Which is simply getting a pointer to the data.
