# Statistical Modeling

1. Hypthesis to investigate Pehonmena to study
2. Design Experiment
3. Collect Data
4. EDA
5. Propose Statistical Model
6. Fit Model
7. Model Validation 
8. (Go back to 5. )
9. Summarize Model Validation
10. Scientific Conclusion

### Mean Square Error

Lets consider the statistical model: Y = g(x) + e

Then the mean squared error is: E((Y - g(x))^2) = bias(g(x))^2 + Var(g(x)) + variance

### Bias  Variance Tradeoff

Higher variance and very low bias corrsponds to overfitting, lower variance but high bias correpsonds to underfitting. Mean square error encodes both and a minimum of the MSE is desirable.

___



## Resampling

### Empirical density of the estimator

1. Take many samples from the population
2. Compute the estimator
3. Analyze / Plot the empericial density of the estimator

```{r}
N = 1000
vars = numberic(N)
for (i in c(0:N)) {
    samples = rnorm(n = 5, mean = 0, sd = 1)
    vars[i] = var(samples)
}
# Histogram with 50 breaks
hist(vars, breaks = 50)
```

### Cross validation

- **Holdout:** Simple test and training split
- **K-fold:** divide dataset into k subsets, then use one of the subsets as test and the others as training sets
- **Leave one out:** Similar as K-fold but in this case k = N.

### Bootstrapping

Take a sample from the sample.

### Parametric Bootstrapping

1. Estimate parameters of population distribution
2. Use the estimates to simulate the samples

Example on how to generate an universally usable test and train split:

```r
sampleSize = 0.7 * n
# Seq_len creates a sequence which starts at 1 and ends at the given value
train_ind = sample(seq_len(nrow(mtcars)), size = sampleSize)
train <- mtcars[train_ind, ]
test <- mtcars[-train_ind, ]
```



## PCA

Create new independant variables where the new independant variables each is a combination of old independant variables. 

We want to maximize the variance on the selected axis. 

```{r}
# scaling will scale the variables to have a unit variance before performing the PCA
pca = prcomp(data, scale = TRUE)
# Gives a summary of the pca
# sdev are the standard deviations of the prinipal components
# rotation, contains the matrix with the eigenvectors
# center The varibale means
# scale the variables standard deviations => the scaling applied to each variable
# Coordinates of the individual observarions
str(pca)
# Plot of the amount of variance explained by each PC
screeplot(pca)
plot(pca)
# Explaines how the two PC are being constructed based on the original variables
biplot(pca)

```

### How many features should we keep?

*Method 1* take as many features s.t. we can explain 75% of the variance. 

*Method 2* Identify the ellbow in the screeplot

### When is PCA suitable

- Multiple continous variables
- Linear relationship between all values
- Large samples are required
- Adequate correlations to be able to reduce variables to a smaller number of dimensions
- No siginifacant outliers

### EOF 

Empirical Orthogonal Functions is when we apply a PCA onto for example a grid. Can be done identical, we simply need to transform the data from a 2D grid data into 1D array. 

___



## Clustering

### Hierarchical Clustering

Dissimilarity can be defined with different strategies. 

- **Single Linkage:** (Nearest neighbour linkage) will measure the distance between the two nearest samples from two groups.
- **Complete Linkage:** (Furhest Neighbout linkage) will measure the distance between the two furthest samples from the two groups. 
- **Ward Method:** 

### K-Means Clustering





### Model Based Clustering

