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

Profile:

Install Intel® VTune™ Profiler for Linux* OS 



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

#### Double pointers

````c++
int *pointer;
int *doublePointer = &pointer;
````

TODO: Why on earth do we need double pointers?? 

#### Void pointers

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



#### Unique pointers

TODO

#### Vectors

````c++
int var;
int arr[100];

````



### Refereneces

Now that we have pointers, why do we still need references?

```c++
// passes class Bar by reference and not by value
void foo(Bar & b) {
    return b.x;
}
```

Usually a pointer is pretty much the same as a reference. The main difference is that we do not need to use the derefence operator.



## Memory allocation

```c++
// static allocation, rather autmomatic allocation
int number = 88;
int * p1 = &number;

// dynamic allocation
int * p2;
p2 = new int;

// remove dynamically allocated space
delete p2;

// this also works for more complex types
T * p3;
p3 = new T(<arg>)
```

### Memory Leaks

the infamous memory leaks.

Happens when we use ``new`` to allocate memory, but we do not use ``delete`` to deallocate the memory. 

But does this also happen when we use static memory allocation?

## Bit operations

````cpp
// Shift
int a = 16;
int b = a << 1;
// b = 8
````



## L and R values

https://www.internalpointers.com/post/understanding-meaning-lvalues-and-rvalues-c

temporaries are Rvalues

```c++
int x = 666; 
// 666 is an r value. Has no specific memory address, stored on temporary register. 
// x on the other hand is an l value, it does have a specific memory location. 
```

### Rvalue references and move constructors

A move constructor can take another datastructure and move its content without creating a deep copy. More 

```c++
std::auto_ptr <int> b (new int (10));
std::auto_ptr <int> a (b);
```



## Object Oriented

### Structs

Whenever possible use a struct instead of a class.

```c++
struct Bar {
    int a;
}
```

Structs can also have constructors:

```c++
struct Bar {
    Bar(int b ) : a(b) {
        
    };
    int a;
}
```

and functions: 

```c++
struct Bar {
    Bar(int b ) : a(b) {
        
    };
    int a;
    
    void foo(int c) {
        a += c;
    } 
}
```

Member functions should have the const keyword, which provides a function signature, when they are simple getters, i.e. do not modify the underlying data:



```c++
struct Bar {
    int a;
    
    int getA() const {
        return a;
    } 
}
```

In some cases we cannot make use of classes, when for example: (When exactly?)

### Classes

We can define an abstract class, usually in the header file as follows:

```c++
class Bar {
    virtual void foo() = 0;
}

```

This function needs to be implemented in another header file which derives Bar also there needs to be a function definition. 



### (Pure) virtual functions

```c++
class A {
    // a virtual function, need to be redefined by derived class
    virtual void v();
}
```

```c++
class B {
    // a pure virutal function, makes B an abstract class, means B cannot be instantiated
    virtual void pv() = 0;
}
```



## Types & casting

Implicit conversion:

```c++
int i = 10;
float a = i;
```

Explicit conversion (Traditional):

```c++
int i = 10;
float a = (float) i;
```

#### More explicit explicit conversions:

Dynamic Cast, only used on pointers and references to objects. Makes sure that type conversion is valid.

```c++
class CBase { };
class CDerived: public CBase { };

CBase b; CBase* pb;
CDerived d; CDerived* pd;

pb = dynamic_cast<CBase*>(&d); // ok
pd = dynamic_cast<CDerived*>(&b); // error
```

Static Cast, does not perform any safety checks. Thus programmer has to make sure that conversion is valid.

```c++
class CBase {};
class CDerived: public CBase {};
CBase * a = new CBase;
CDerived * b = static_cast<CDerived*>(a) // ok
```

Reinterpret Cast, performs a simple binary copy. Can cast anything to anything.

```c++
class A {};
class B {};
A * a = new A;
B * b = reinterpret_cast<B*>(a);
```

Const cast, can convert a const into a non const.

```c++
const int a = 10;


void foo(int * x) {
    return;
}

foo(const_cast<int *>(a));
```

### Typedefs

Defines a typedef declaration.

This typedef defines a type for an unsigned long.

```c++
typedef unsigned long ulong;
```



## Templates

```c++
template<class T>
class Bar {
public:
    T b;
    Bar(T a) {
        b = a;
    }
}
```

Or several explicityl defined template classes:

```c++
template<class T, class X>
class Bar {
public:
    T b;
    Bar(T a) {
        b = a;
    }
    
    X foo(X c) {
        return X;
    }
}
```

Or applied to functions:

```c++
template<class T> void f(T a) {
    std::cout << a << std::endl;
}
```



### Variadic Templates or template packs

```c++
template<class ... Ts> void f(T a) {
    std::cout << a << std::endl;
}

```

TODO



## Circular Dependencies

Yes they are annoying. 

