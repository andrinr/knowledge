# FORTRAN

## Common Pitfalls

- Arrays start at 1 not 0



## Structure

````fortran
module module_name:
	! used modules
	use module_name_b
	...
	
	! declarations
	integer :: var3, var4 = 3
	...
	
contains

	subroutine subroutine_a
	
		
	end subroutine_a
		
	
	subroutine subroutine_b
	
		! do stuff
		
	end subroutine_b
	
end module module_name
````



## Declaration

````fortran
!integer
integer :: var1, var2
! initialization
integer :: var3, var4 = 3
! boolean
logical
! array with fixed dimension
integer, dimension(2) :: (/0,0/)

! disable implicit variables, meaning all varibales need to be redeclared
implicit none
````



## if / else

```fortran
if (var1 == var2) then 
	! do stuff
end if

! or
if (var1 .EQ. var2) then
	! do stuff
end if

! Not equal
if (var1 .NEQ. var2) then
	! do stuff
end if

! if else
if (...) then
	! do stuff
else
	! do stuff
end if
```


## Loops

````fortran
do i=<start>,<end>
	! do stuff
end do
````



## Calling subroutines

````fortran
call subroutine_name

! or with parameters
call subroutine_name(param1, param2)
````



## Multiline statements

````fortran
! use & to connect multiline statements .i.e.
call subroutine_name(&
	param1,&
	param2&
)
````

