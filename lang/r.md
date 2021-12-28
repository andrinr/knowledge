# r

## Data structures

```{r}
# Array
a = c(1:3)
b = c(4:6)
c = c(a, b)

# Convert the data into a datastructure of given sizes in corresponding dimensions
array(data, c(4, 5))

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
means = apply(x, 2, FUN = function(x) {
    mean(x)
})
```



## Linear regression

```{r}
lr = lm(response ~ predictor1 + predictor2, data=data)
summary(lr)
```

Lets consider this example:

```{r}
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -29.069  -9.525  -2.272   9.215  43.201 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  42.9800     2.1750  19.761  < 2e-16 ***
## speed.c       3.9324     0.4155   9.464 1.49e-12 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 15.38 on 48 degrees of freedom
## Multiple R-squared:  0.6511,	Adjusted R-squared:  0.6438 
## F-statistic: 89.57 on 1 and 48 DF,  p-value: 1.49e-12
```

Residuals are the difference between the actual observed response variables and the predicted response variables. This can further be examined with the residuals plot. 

Coefficients: We have the four columns, where the Estimate is the actual estimate for the coefficent. The standart error which defines the expected variance from the value if we where to rerun the experiment. The t value, the higher it is, the more likely we can reject the null hypothesis. Finally the p value should be smaller than 5%. 

The residual standard error is the average term of the residuals. 

The multiple r-squared which desribes the percentage of the variance which can be explained with the model. 

F statistics: the further away from 1, the better (larger is better). 

