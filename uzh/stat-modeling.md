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



