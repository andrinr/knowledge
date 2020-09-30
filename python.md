# Python

### Syntax and best practices

- **Don’t** use camel case, use snake case i.e. ``this_is_a_function``
- Two lines of spacing before each function

#### for loops

A simple for loop:

````pyt
for i in range(100):
	# do stuff
````

#### functions

````py
def function():
	# do stuff
	return
````

## Numpy

### Arrays

`````py
 np.linspace(start, end, count)
`````





## Matplotlib

### Common Pitfalls:

- Don’t forget ``plt.show`` to show the plot(s)

### Subplots

````py
fig, axs = plt.subplots(2, 2)
axs[1,1].scatter(...)
````

### Animation

````python
FuncAnimation(fig, update, frames=[...], interval=X, repeat=True)
````

update function is called in each frame. Basically you want to plot before calling the FuncAnimation and then alter the plot or its data in the update function. For example one can do:









## Pandas







