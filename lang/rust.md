# Rust

- https://doc.rust-lang.org/stable/rust-by-example/
- https://doc.rust-lang.org/book/

## Style Guide

- We use snake case for all variables and functions
- We add a _ for varibales which are unused. 
- We use UpperCamelCase for type aliasing. 
- const ALL_UPERCASES

## Ownership

### The Stack and the Heap

- Stack stores issentially a last in, first out datastructure
- Data on the stack must have known fixed size, data on the heap must not
- The stack is generally faster because the allocator does not need to search for a place to store new data

### Ownership Rules

- Each value in Rust has a variable that’s called its *owner*.
- There can only be one owner at a time.
- When the owner goes out of scope, the value will be dropped.

### With the example of strings

- We have two types of strings: String and &str
- A String is a vector of UTF-8 sequences. String is heap allocated and this growable and supports operators such as push.
- A &str is the same as an array of UTF-8 sequences. Its fixed in size.

Memory for a String must be allocated at runtime. 

In Rust we have no GC and memory is auto freed as soon as the variable gets out of scope.

To ensure memory safety something like this is not possible:

```rust	
let a = String::from("hello");
let b = a;
println!(a); // will return an error

let e = String::from("hello2");
take_ownership(e);
// From this point on, e is no longer available and was freed because the function took ownership
fn take_ownership(some_string : String) {
    ()
}

let k = 7;
let j = k;
println!(k); // This however works since the type of a is an integer which has a fixed size and can be stored in the stack
```

We can however also copy the heap data of a:

```rust
let b = a.clone();
```

### References and Borrowing

A reference essentially points to a stack entry, which then again points to a heap entry. This process is called borrowing.

```rust
let e = String::from("hello2");
not_take_ownership(&e);
// From this point on, e is no longer available and was freed because the function took ownership
fn not_take_ownership(some_string : &String) {
    ()
}
```

Note that we can not modify borrowed items as we do not have ownership over them. However we can modify the code such that we have a mutable reference: 

```rust
let mut e = String::from("hello2");
not_take_ownership(&mut e);
// From this point on, e is no longer available and was freed because the function took ownership
fn not_take_ownership(some_string : &mut String) {
    ()
}
```

There can hoever only ever exist one mutable reference to particular piece of data at a time. This will prevent data races at compile time. 

```rust
let mut s = String::from("hello");

let r1 = &s; // no problem
let r2 = &s; // no problem
println!("{} and {}", r1, r2);
// variables r1 and r2 will not be used after this point

let r3 = &mut s; // no problem
println!("{}", r3);
```

Note that ownerships ends as soon as a varibale is not used anymore.

All of this prevents dangling references, a reference to memory where we may have different data than we originally intended to be.

## Primitives

```rust
// Types
let a = 32; // default type
let b = 32.0; // default type
let c: f64 = 32.0; // type annotation

// Tuples
let d : (i32, bool) = (1, false);
```



### Arrays, Vectors and slices

```rust
// Arrays: fixed sized lists, per default immutable 
let xs : [i32, 4] = [0, 1, 2, 3]; // explicit
let ys : [i32, 4] = [0; 4]; // constant
&xs[1..3]; // get slice of an array

// Borrow the array as a slice
&xs
// or just a part of it
xs[1..3]

// Vectors: Growable list 
let v = vec![1, 2, 3] // the same as vec!(1, 2, 3)

```



### const vs Non-mut lets

- Const must have their types annotated
- Const can be in any scope
- Const can only be set to a constant expression, not a result computed at runtime

Both const and let without a mut define immutable variables. The difference is that const is a compile time evalution while let is a runtime evaulation. A let is a memory location, while a const is not.

## Block Expressions

```rust
let y = {
    let x_squared = x * x;
    let x_cube = x_squared * x;

    // This expression will be assigned to `y`
    x_cube + x_squared + x
};

let big_n =
    if n < 10 && n > -10 {
        10 * n
    } else {
        n / 2
    };
```

## Loops

```rust
// An infinite loop
loop {
    continue;
    break;
}

// Break outer loop directly
'outer: loop {
    println!("Entered the outer loop");

    'inner: loop {
        println!("Entered the inner loop");

        // This would break only the inner loop
        //break;

        // This breaks the outer loop
        break 'outer;
    }

    println!("This point will never be reached");
}

// Return value after break
let result = loop {
    counter += 1;

    if counter == 10 {
        break counter * 2;
    }
};
```

## Match (Switch)

```rust
 match number {
     // Match a single value
     1 => println!("One!"),
     // Match several values
     2 | 3 | 5 | 7 | 11 => println!("This is a prime"),
     // TODO ^ Try adding 13 to the list of prime values
     // Match an inclusive range
     13..=19 => println!("A teen"),
     // Handle the rest of cases
     _ => println!("Ain't special"),
     // TODO ^ Try commenting out this catch-all arm
}
```

## Functions

```rust
// Implementation block, all `Point` associated functions & methods go in here
struct Point {
    x : f64,
    y : f64,
}
impl Point {
    fn origin() -> Point {
        Point { x: 0.0, y: 0.0 }
    }
    fn new(x: f64, y: f64) -> Point {
        Point { x: x, y: y }
    }
    // Access data with self
    fn length_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }
    // translate 
    fn translate_x(&mut self, x : f64) {
        self.x += x;
    }
}

// acess like
let a = Point::origin()
a.translate_x(0.5);

// Functions which does not return anything
fn foo() -> ! {
    panic!("This call never returns.");
}
```

## Higher Order Functions

TODO

## Pointers

