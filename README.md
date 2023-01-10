# RegressionModels
RegressionModels is a Julia package of solving regression problem: 

Given a sequence of samples ```(X_j)``` and a sequence of values ```phi(X_j)``` of some unknown function ```phi```, find the value```phi(a)``` for any given point ```a```.

To tackle this problem, we utilize polynomial regression.


# Required softwares
RegressionModels has been implemented on a desktop compute with the following softwares:
- Ubuntu 18.04.4
- Julia 1.7.1
- [Mosek 9.1](https://www.mosek.com)


# Installation
- To use RegressionModels in Julia, run
```ruby
Pkg> add https://github.com/maihoanganh/RegressionModels.git
```

# Usage
The following examples briefly guide to use RegressionModels:

## Regression

```ruby

r=1.7
nr=100 # number of samples
X = zeros(Float64,nr,1)
for j in 1:nr
    X[j,1] = 2r*rand()-r
end

phi(a)=sin(a)+0.2*cos(5*a) # exact relation of the input and output
eta = Vector{Float64}(phi.(X[:,1])) # exact output for the samples

# devide the samples into train set and test set

t=Int64(0.8*nr)
X_train=X[1:t,:]
eta_train=eta[1:t]

X_test=X[t+1:end,:]
eta_test=eta[t+1:end]

println("Number of samples: ", nr)
println("Cardinality of train set: ",t)
println("Cardinality of test set: ",nr-t)

using RegressionModels


n=1 # number of attributes
c=Inf # parameter for bound constraints
k=3 # degree of polynomial approximation

eval_pol_approx=RegressionModels.model_pol_regress(n,X_train,t,c,k,eta_train,additional_monomials=false)

predict=[eval_pol_approx(X_test[j,:]) for j in 1:length(eta_test)]

using LinearAlgebra

error=norm(predict-eta_test)/maximum([norm(predict);norm(eta_test)])

println("Error: ", error)
```

See other examples from .ipynb files in the [link](https://github.com/maihoanganh/RegressionModels/tree/main/test).


# References
For more details, please refer to:

**N. H. A. Mai. Convergent hierarchies of polynomial approximations for regression and classification. 2023. Forthcoming.**

To get the paper's benchmarks, download the zip file in this [link](https://drive.google.com/file/d/14yxm858LhCMkTCZopNlGDkqrUgMiJYwP/view?usp=sharing) and unzip the file.

The following codes are to run the paper's benchmarks:
```ruby

data="/home/hoanganh/Desktop/math-topics/algebraic_statistics/codes/datasets" # path of data 
#The path needs to be changed on the user's computer

using RegressionModels

RegressionModels.test()

RegressionModels.test_univariate_regression(data) # Figure 1, first subfigure
RegressionModels.test_univariate_regression2(data) # Figure 1, second subfigure
RegressionModels.test_univariate_classification_pol(data) # Figure 2, first subfigure
RegressionModels.test_univariate_classification_pol2(data) # Figure 2, second subfigure

RegressionModels.test_house_regression(data,1) # Table 5, degree 1
RegressionModels.test_house_regression(data,2) # Table 5, degree 2
RegressionModels.test_house_regression(data,3) # Table 5, degree 3

RegressionModels.test_Breast_cancer_polynomial_regression(data,1) # Table 10, degree 1
RegressionModels.test_Breast_cancer_polynomial_regression(data,2) # Table 10, degree 2

```
