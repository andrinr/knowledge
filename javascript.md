# js / ts

## Hash Maps

````typescript
protected elements: Map<key, Object> = new Map();

// Iterate over map:
hashMap.forEach(
	(value, key) => {
        // do stuff 
    }
);

// delete element 
hashMap.delete(<key>);
               
// add element
hashMap.set(<key>,<object>);

// get element
hashMap.get(<key>)
````



## Animation

````typescript
// Basic animation loop
run(){
    // Do stuff within loop
    window.requestAnimationFrame(this.run);
}

````

### Imports

Some packages cannot be imported the regular way because they depend on the ``document`` element. They then need to be imported inline. 

````javascript
const {default: Isotope} = await import('isotope-layout');
// With rollup make sure to set this flag to true
inlineDynamicImports: true
````



## VUE



## REACT



## SVELTE





