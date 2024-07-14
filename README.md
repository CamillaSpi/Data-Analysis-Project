# High dimensional space Regression and Logistic Classifier over time
This project is composed of two different tasks.
<div align='center'>
  <img src="https://static0.srcdn.com/wordpress/wp-content/uploads/2019/03/Game-of-Thrones-Season-8.jpg" alt="GoT" style="width:50%; max-width:400px;">
</div>

# First Task
## Overview
Starting from a data set consisting of n = 70 observations of a dependent variable Y and p = 50 regressors Xj (j = 1 = 2 . . . , p) potentially useful for predicting Y, it was necessary, after comparing different regression techniques, to identify the linear model that minimises the prediction error on a test set and to estimate the coefficients of the independent variables
significant for the prediction of Y, using the 80% of the dataset for the training and the other 20 % for the test. The result obtained on this task supports the next one if the estimated values of the βj coefficients are divided by 100 and rounded to the nearest integer. The integers thus obtained represent the ASCII decimal codes of alphanumeric characters which, ordered by increasing values of j, will form a string
which will represent a clue to the solution of the second task.
## Development
The problem is quite high-dimensional with a number of predictors that is similar to the number of samples in
the training set. The high-dimensional space
exposes a model to fit extremely on the training data, reducing the bias error but increasing the variance.
Apart from this, an operation really complex, in a small set of data like this, is the detection of the outliers.
Another possible problem to take care of is the correlation between the predictors; in a high dimensional problem it could be dangerous to fit a model. So, all these aspects have been analysed and the techniques used are chosen to fit this type of problems. In particular it was tried to apply techniques that are commonly used in predictive modeling, especially in case of high dimension problems for their robustness through addition of bias terms. We have analyzed for each of them the characteristics in their formulation and the results that they obtain in our instance. We have conducted different studies in order to understand which approach is the most effective in modelling the relationship between the independent variable and the dependent ones. First three applied methods, **Ridge Regression**, **Lasso Regression** and **ElasticNet Regression**, are regularization techniques that add a penalty term to the residual sum of errors loss function. This penalty term shrinks the coefficients towards zero. The degree of shrinkage can be controlled by a hyperparameter called the regularization strength (lambda), which determines the trade-off between model fit and simplicity. 
It was also applied **Subset selection with AIC and BIC**, in order to perform variable selection with an estimate of the test error, thus with mathematical adjustment techniques on the training error, not actually using the test samples.
Finally it was also done an evaluation of the performance of **Linear Regression through OLS**, knowing its limitations in this environment, but only to have a benchmark for evaluations.
To evaluate the performance of all these described techniques it was compared the variable selection produced, and the generalization performance of each approach. The variable selection can be evaluated by comparing the magnitude of the coefficients and the number of non-zero coefficients for each approach. Predictors with low p-values and high regression coefficients are likely to be the most significant contributors to the model. The generalization performance can be assessed by splitting the data into training and test sets, and by comparing the prediction performance on the test set for each approach. 

By comparing the performance of these techniques, it was possible to gain insight into the strengths and weaknesses of each approach and so starting from the obtained results it is possible to conclude that the best technique for the specific data analysis problem is **Lasso**, that is the one performing better in terms of MSE, and also because, observing the correlation matrix, it is clear that only a small subset
of the regressors is really significant for the response, and so an approach that performs variable selection is required. 

The suggestion of this first task turns out to be: *"GoT"*.

For more details on design choices, experiments conducted, results obtained and explanatory diagrams, please refer to the [Report](Report Final Exam Data Analysis Group 7.pdf).

# Second Task
## Overview
Starting from a training set obtained from a stream of data collected at successive time instants n = 1, 2, . . . , N, and consisting of the feature-label pairs (X1, Y1),(X2, Y2), . . . ,(XN , YN ), where Xn is d-dimensional and Yn ∈ {-1, 1}, it was necessary to make an appropriate reduction in the dimensionality of the training set. With such a reduced dimensionality training set it was then necessary to train a classifier by applying the appropriately implemented stochastic gradient algorithm with constant step-size, evaluating its behavior for different step-size choices. The system thus implemented makes predictions at specific time instants t1, t2,....,tK, by observing
the features Xtest(1), Xtest(2),...., Xtest(K) and applies a decision rule such that it returns +1 if the product between the reduced dimensionality feature and the parameter estimated by the stochastic gradient algorithm at time j-th is greater than zero, -1 otherwise.
It was then necessary to convert the binary string obtained by classifying the observations into ASCII characters encoded with 8 bits, associating bit 0 with the value -1. The characters obtained are taken from a famous sentence related
to the clue obtained in the first task.
## Development
First of all, an exploratory analysis on the dataset has been conducted, considering the fact that it is made by
different acquisitions over time, in order to evaluate the the trend of the features over time and it was clear that the data seems to be not stationary as a whole, but rather stationary within a specific range of time intervals. Next, dimensionality reduction (PCA) was applied through a manual implementation of eigen decomposition on scaled data. To choose the number of principal components, it was evaluated what could be a good tradeoff between the decrease in dimensionality and the explained variance. Obviously, the PCA projects the data in a new feature space in which each component is a linear
combination of starting ones. So, to ensure that the hypothesis made on the starting features are preserved in this new space it was repeated the analysis also on the obtained principal components. Surely, it was expected that the trend would be preserved and also that the values fluctuations over time should be emphasized on the firsts principal components because they enclose the most of data variations. The SGD algorithm with constant step size was implemented, in order to be more able to change with a truth variation like that happens through time, trying different values. Performing the test procedure it was obtained as result: **"WinterComing"**.
Further analyses were also conducted, such as: 
- evaluation of the instantaneous error committed during the training procedure for different values of principal components and step sizes;
- variation of the coefficients over time;
- comparison over time of actual values with predicted values and relative confidence;
- comparison with a simple Naïve Bayes approach, which requires a series of assumptions to be applied, analysing the separability of the features, concluding that even just two or one of the features could be sufficient to discriminate the classes, obtaining again correct predictions.

For more details on design choices, experiments conducted, results obtained and explanatory diagrams, please refer to the [Report](Report Final Exam Data Analysis Group 7.pdf).


## Repository Structure
- `First Task/`: contains the codes and a brief report on the obtained results for this task;
- `Second Task/`: contains the codes and a brief report on the obtained results for this task;
- `Report Final Exam Data Analysis Group 7.pdf`:  is the final report containing the description of design choices, experiments conducted, results obtained and explanatory diagrams.


## Feedback
For any feedback, questions or inquiries, please contact the project maintainers listed in the Contributors section.
