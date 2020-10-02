# Python

## Syntax and best practices

- **Don’t** use camel case, use snake case, for example: ``this_is_a_function``
- Two lines of spacing before each function

#### for loops

A simple for loop:

````python
for i in range(100):
	# do stuff
````

#### functions

````python
def function():
	# do stuff
	return
````





## Numpy

### Arrays

`````python
# generate linear array 
np.linspace(start, end, count)

# 2D Array
np.array([1,2,3],[4,5,6],[7,8,9])

# Array shape
array.shape

# Init array of any shape with zeros
numpy.zeros((x,y,z))

# Access entry in n-dim array
array[x,y,z]

# Negative indexing, can be used to select n last elements
# last element
array[-1]
# last n elements
array[-n]

# Slicing
array[start:end]
# With steps
array[start:end,step]

# Flatten array i.e. [[1,2],[3,4]] => [1,2,3,4]
array.flatten()
`````





## Matplotlib

### Common Pitfalls:

- Don’t forget ``plt.show`` to show the plot(s)

### Subplots

````python
fig, axs = plt.subplots(2, 2)
axs[1,1].scatter(...)

# Axis title
axs[1,1].set_title("title")

# Add spacing between subplots
fig.tight_layout(pad=3.0)
````

### Animation

````python
FuncAnimation(fig, update, frames=[...], interval=X, repeat=True)
````

``update`` function is called in each frame. Basically you want to plot before calling the ``FuncAnimation`` and then alter the plot or its data in the update function. For example one can do:





## Pandas





## Seaborn









