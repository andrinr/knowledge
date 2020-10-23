# NPM

````bash
npm init

npm version x.x.x
# define a beta or alpha version
npm version x.x.x-beta.x

npm publish
# publish a beta or alpha
npm publish --tag beta

# unpublish
npm unpublish <packagename>@<version>
````



## package.json

````json
// run stuff concurrently 
'scripts' : {
    "test" : "npm run tslin & npm run test"
}
// run stuff sequentially
'scripts' : {
    "prep" : "npm run test && npm run build"
}
````



