# Rust

https://doc.rust-lang.org/stable/rust-by-example/

## Style Guide

- We use snake case for all variables. 
- We add a _ for varibales which are unused. 
- We use UpperCamelCase for type aliasing. 



## Primitives

```rust
// Types
let a = 32; // default type
let b = 32.0; // default type
let c: f64 = 32.0; // type annotation

// Tuples
let d : (i32, bool) = (1, false);

// Arrays
let xs : [i32, 4] = [0, 1, 2, 3]; // explicit
let ys : [i32, 4] = [0; 4]; // constant
&xs[1..3]; // get slice of an array

```

### const vs Non-mut lets

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

