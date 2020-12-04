# CSS

## ::after

Useful selector to create pseudo class which mimics the properties of its parent. 

````css
.class::after{
    background-color: 'red';
}
// Background element
// Make sure the position of the parent element is set to relative (!)
.class::after{
    position: absolute;
    width: 100%;
    height: 100%;
    top: 0;
    left: 0;
}
// Add hover effects to ::after
.class:hover::after{
    // stuff
}
````

