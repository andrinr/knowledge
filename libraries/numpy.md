# Numpy

## Arrays

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



