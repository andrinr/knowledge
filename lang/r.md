# r

## Data structures

```{r}
# Array
a = c(1:3)
b = c(4:6)
c = c(a, b)

# Initialize without values
d = Numeric(100)

# Matrix, by default filled by columns
matrix(c(1:9), nrow=3)
# 1 2 3 4 5 6 7 8 9
# ==> 
# 1 4 7 
# 2 5 8
# 3 6 9
```



## Apply

```{r}
# Lets say n means of samples of size m
X = matrix(data = rnorm(n * m, mean = 0, sd = 1), nrow = n, ncol = m)
# margin (second arg) 1 = rows, 2 = cols, in this case we want to take the means over the columns
means = apply(x, 2, )
```



## Linear regression

```{r}
lr = lm(response ~ predictor1 + predictor2, data=data)

summary(lr)
```



