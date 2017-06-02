dataset = read.csv('Salary_Data.csv')

library(caTools)

split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

regressor = lm(formula = Salary ~ YearsExperience, data = training_set)
#summary(regressor)

y_pred = predict(regressor, newdata = test_set)

# install.packages('ggplot2')
library(ggplot2)

visualize_result <- function(X, y, title, xlabel, ylabel){
  ggplot() +
    geom_point(aes(x = X, y = y),
              colour = 'red') +
    geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
              colour = 'blue') +
    ggtitle(title) +
    xlab(xlabel) +
    ylab(ylabel)
}

simple_linear_regression_visualization <- function(X, y){
  visualize_result(X, y, 'Salary vs Experience', 'Years of experience', 'Salary')
}

visualize_result(training_set$YearsExperience, training_set$Salary)
visualize_result(test_set$YearsExperience, test_set$Salary)