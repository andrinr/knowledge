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

### QA

- PCA can be used for projecting and visualizing data in lower dimensions: T
- Max number of PC’s = number of features: T
- All PC’s ar orthogonal to each other: T
- What happens when eigenvalues are roughly equal? PCA will perform badly

___



## Clustering

### Hierarchical Clustering

Dissimilarity can be defined with different strategies. 

- **Single Linkage:** (Nearest neighbour linkage) will measure the distance between the two nearest samples from two groups.
- **Complete Linkage:** (Furhest Neighbout linkage) will measure the distance between the two furthest samples from the two groups. 
- **Ward Method:** In each step find the pair which leads to minimal increase in total within cluster variance.

```{r}
# hclust requries a dissimilary structure, in this case we will go with dist which computes the eulerian distance between all the rows of the input matrix
hc1 = hclust(dist(matrix), method = "single")
hc2 = hclust(dist(matrix), method = "complete")
hc3 = hclust(dist(matrix), method = "ward.D")

# We can plot the clustering like this:
plot(hc1)

# We can return a datastructure where each item is assigned to a cluster, provided number of clusters
cut = cuttree(hc1, 6)

# We can do the same visually like this:
par(mfrow = c(1, 1))
plot(hc1, xlab = "", sub = "")
rect.hclust(hc1, k = 5, border = rainbow(6))
```



### K-Means Clustering

1. Start with k (random) cluster centers
2. Assign obervations to the nearest center
3. Recompute the centeres
4. If centers remain the same, stop, otherwise repeat step 2.

### Model Based Clustering

We assum that the data is a mixture of several clusters, this means that there are soft border between them. 

### Assesing Quality

Is unsupervised therefore measuring the quality is difficult. Cluster can easily be found in random data, and sometimes its hard to distinguish these from meaninungful clusters.

- Shilouette plots indicate the nearest distance of each sample to the another cluster. Distances towards 1 are favourable, 0 or even negative may indicate wrong assignments. 

### QA

- Two runs of K-Means will have the same results: F
- It is possible that the K-Means assignment does not change between two iterations: T
- Clustering is unsupervised: T
- K-Means automatically chooses an optimal number of clusters: F
- Hier. Clustering depends on the linkeage method: T

## Classification

### Linear Discrimination

We assume that both categories have a gaussian distribution. 

In the linear case we assume for the distributions to have the same covariance matrices, only the means can vary. We have several distributions and we want to associate each sample to a distribution. The distributions are known, but the samples are intertwined, thus we have to use LDA to decide which sample can be associated with which distribution. 

The discrimination line is at the point (in 2D) with the same densities.

```{r}
lda = lda(response ~ predictor1 + predictor2, data = data)
```



### Quadratic Discrimination

In the quadratic case the distributions can have varying variances. In fact for all samples both variants can be applied but QDA performs than LDA with different variances.

```{r}
qda = qda(response ~ predictor1 + predictor2, data = data)
```

### Fishers Discriminant Rule

It is identital to LDA when covariance matrices are equal. 

### Classification Tree

Actually pretty much the same as a K-D tree, but in this case we have more than 3 Dimensions.

- Easy to understand, explain and visualize. 
- Non robust method, 

```{r}
ct = ctree(response ~ ., data = data)
# Plot it
plot(ct)
```



### Bagging

Use bootstrapping to create many training sets and for each of them create a new tree. Take the average for the final classification.

### Boosting

Similar to bagging but samples are associated with a weight which corresponds to the amount of missclassifications, with this the training will focus on hard cases.

### Random Forest

Similar to bagging but for each tree we choose a random subsample of the featutres, where k = sqrt(n). 

```{r}
rf = randomForest(response ~ ., data = data)
plot(rf)

# Optionally repeat rd with only most important features extracted with
varImpPlot(rf)
rf2 = randomForest(response ~ a + b + c)
```

### QA

- LDA will fail when the discriminatory information lies in the variance and not the mean: T
- LDA tries to model difference in classes, PCA does not take any differences into account: T
- In RF individual trees are buit on a subset of features and observations: T
- Given one very strong predictor, would you rather use bagging or random forest? RF
- 

## Linear Model

```{r}
# Classical linear model
lm(predictor ~ responseA + responseB, data = data)
# Linear model with explicitly defined intercept term
lm(predictor ~ 1 + responseA + responseB, data = data)
# Linear model with formulate applied to response variable
# Note that the I() is necessary to bypass the meaning of operators in the context of the LM formula
lm(predictor ~ responseA + I(responseB * 2))
```



### Assumptions

- **Homoscedastic:** Var(error_i) = variance. Look at scale location plot, watch out for a horizontal line and equally spread points. 
- **Independence:** Corr(error_i, error_j) = 0 for i not equal j
- **Gaussian:** The errors are gaussian with a mean of 0. We can look at the QQ plot, if we see a more or less linear distribution, we can assume that the error are gaussian distributed.
- **Linearity:** The regression model is linear in parameters. Look at residuals vs. fitted. Equally spread residuals without distinct patterns are a good indicator for a linear relationship.

Residuals versus leverage helps us identify cases which have high leverage.

### Leave on out for linear models

What is the effect of the i-th obersvation on

- the estimate
- the prediction
- the estimated standard errors

````
lm = lm(predictor ~ response1 + response2, data=data)
influence.measures(lm)
````

### ANOVA

Analysis of variance

```{r}
anova(lmA, lmB)
```



### Information Criterion

Balances goodess of a fit with its complexity. 







