# GDB

1. Compile with -g flags for better info
2. run with gdb flag to debug
3. type ``run``
4. type ``backtrace`` to trace error

## Printing

1. ``print <varname>`` to get value of variable
2. ```p *&p[<start>]@<count>``` to print a range of an array (p must be a pointer / c style array)

## With MPI

``mpirun -np <x> gdb <exec> <params>``

Or with a console for each process:

``mpirun -np <x> xterm -e gdb <exec> <params>``

Or run gdb on only one process:

``mpiexec -n 1 gdb <exec> <params> m : -n <x-1> <exec> <params>``

