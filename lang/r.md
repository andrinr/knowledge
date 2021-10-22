# r

## Data structures

```{r}
# Array
a = c(1:3)
b = c(4:6)
c = c(a, b)

# Matrix
matrix(c(1:9), nrow=3)
```



## Linear regression

```{r}
lr = lm(response ~ predictor1 + predictor2, data=data)

summary(lr)
```



