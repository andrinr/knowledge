#  Matplotlib

## Common Pitfalls:

- Don’t forget ``plt.show`` to show the plot(s)

## Subplots

````python
fig, axs = plt.subplots(2, 2)
axs[1,1].scatter(...)

# Axis title
axs[1,1].set_title("title")

# Add spacing between subplots
fig.tight_layout(pad=3.0)

# Legends
axs[1].legend(['Euler', 'Kutta'], loc=1)

# Aspect ratio
axs[1].set_box_aspect(1)

# limits
axs[1].set_ylim(-3, 3)

# axes
axs[1].set_xlabel('q')
````

## Animation

````python
FuncAnimation(fig, update, frames=[...], interval=X, repeat=True)
````

``update`` function is called in each frame. Basically you want to plot before calling the ``FuncAnimation`` and then alter the plot or its data in the update function. For example one can do:

## Pixel Plot

````python
# where data is a 2D np array
plt.imshow(data)
````

