dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

library(caTools)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

library(rpart)
classifier = rpart(formula = Purchased ~ .,
                   data = training_set)
y_pred = predict(classifier, newdata = test_set[-3], type = 'class')
cm = table(test_set[, 3], y_pred)

library(ElemStatLearn)
visualize_result <- function(set, title, xlabel, ylabel){
  X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
  X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
  grid_set = expand.grid(X1, X2)
  colnames(grid_set) = c(xlabel, ylabel)
  y_grid = predict(classifier, newdata = grid_set)
  plot(set[, -3], main = title,
       xlab = xlabel, ylab = ylabel,
       xlim = range(X1), ylim = range(X2))
  contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
  points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
  points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
}

decision_tree_classification_visualization <- function(set){
  visualize_result(set, "Decision Tree Classification", "Age", "EstimatedSalary")
}
decision_tree_classification_visualization(training_set)
decision_tree_classification_visualization(test_set)

plot(classifier)
text(classifier)