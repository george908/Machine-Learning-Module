import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

df = pd.read_csv("week2.csv", header= None, skiprows=1) # tell it there are no titles so it
                                                        # can read in the first line and tell it
                                                        # to skip the ID line 
print(df.head())
X1=df.iloc[:,0]
X2=df.iloc[:,1]
X=np.column_stack((X1,X2))
y=df.iloc[:,2] 

# (a)(i)
# loop through all of the y values and plot the x1 and x2 value
# depending on whether y is equal to 1 or -1 
plt.scatter(X1[y==1], X2[y==1], color = 'green', marker = '+', label = '+1')
plt.scatter(X1[y==-1], X2[y==-1], color = 'blue', marker = 'o', label = '-1')
# plot everything 
plt.title('Scatter plot of the first vs the second feature')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.legend(title = 'target')
plt.show()

#(a)(ii)
model = LogisticRegression(penalty=None, solver='lbfgs')
model.fit(X, y)
# print the intercept and the coefficients
print('Intercept:', model.intercept_)
print('Coefficient:', model.coef_)

#(a)(iii)
# loop through all of the y values and plot the x1 and x2 value
# depending on whether y is equal to 1 or -1 
plt.scatter(X1[y==1], X2[y==1], color = 'green', marker = '+', label = 'Actual +1', s = 100)
plt.scatter(X1[y==-1], X2[y==-1], color = 'blue', marker = 'o', label = 'Actual -1', s = 100)
# get the y predictions based on the model and map the x values based on the 
# y predictions using x markers so that they are visible 
y_pred = model.predict(X)
plt.scatter(X1[y_pred==1], X2[y_pred==1], color = 'red', marker = 'x', label = 'Predicted +1', s = 100)
plt.scatter(X1[y_pred==-1], X2[y_pred==-1], color = 'orange', marker = 'x', label = 'Predicted -1', s = 100)
# get the intercept and the parameter weights and map the boundary based on it 
intercept = model.intercept_[0]
parameterWeight0, parameterWeight1 = model.coef_[0]
uniformX1Values = np.linspace(-1, +1, 100)
dBoundary = (-intercept -(parameterWeight0*uniformX1Values)) / parameterWeight1
# plot everything 
plt.title('Scatter plot of the first vs the second feature')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.plot(uniformX1Values, dBoundary, 'k--', label = 'Decision boundary')
plt.legend(title = 'legend')
plt.show()

#(b)(i)
SVM0001model = LinearSVC(C=0.001)
SVM0001model.fit(X, y)
print('0.001 model intercept:', SVM0001model.intercept_)
print('0.001 model coefficient:', SVM0001model.coef_)

SVM1model = LinearSVC(C=1)
SVM1model.fit(X, y)
print('1 model intercept:', SVM1model.intercept_)
print('1 model coefficient:', SVM1model.coef_)

SVM100model = LinearSVC(C=100)
SVM100model.fit(X, y)
print('100 model intercept:', SVM100model.intercept_)
print('100 model coefficient:', SVM100model.coef_)

#(ii)
# for 0.001
plt.scatter(X1[y==1], X2[y==1], color = 'green', marker = '+', label = 'Actual +1', s = 100)
plt.scatter(X1[y==-1], X2[y==-1], color = 'blue', marker = 'o', label = 'Actual -1', s = 100)
# get the y predictions based on the model and map the x values based on the 
# y predictions using x markers so that they are visible 
y_pred = SVM0001model.predict(X)
plt.scatter(X1[y_pred==1], X2[y_pred==1], color = 'red', marker = 'x', label = 'Predicted +1', s = 100)
plt.scatter(X1[y_pred==-1], X2[y_pred==-1], color = 'orange', marker = 'x', label = 'Predicted -1', s = 100)
# get the intercept and the parameter weights and map the boundary based on it 
intercept = SVM0001model.intercept_[0]
parameterWeight0, parameterWeight1 = SVM0001model.coef_[0]
uniformX1Values = np.linspace(-1, +1, 100)
dBoundary = (-intercept -(parameterWeight0*uniformX1Values)) / parameterWeight1
# plot everything 
plt.title('Scatter plot of actual vs predicted values when C=0.001')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.plot(uniformX1Values, dBoundary, 'k--', label = 'Decision boundary')
plt.legend(title = 'legend')
plt.show()

# for 1
plt.scatter(X1[y==1], X2[y==1], color = 'green', marker = '+', label = 'Actual +1', s = 100)
plt.scatter(X1[y==-1], X2[y==-1], color = 'blue', marker = 'o', label = 'Actual -1', s = 100)
# get the y predictions based on the model and map the x values based on the 
# y predictions using x markers so that they are visible 
y_pred = SVM1model.predict(X)
plt.scatter(X1[y_pred==1], X2[y_pred==1], color = 'red', marker = 'x', label = 'Predicted +1', s = 100)
plt.scatter(X1[y_pred==-1], X2[y_pred==-1], color = 'orange', marker = 'x', label = 'Predicted -1', s = 100)
# get the intercept and the parameter weights and map the boundary based on it 
intercept = SVM1model.intercept_[0]
parameterWeight0, parameterWeight1 = SVM1model.coef_[0]
uniformX1Values = np.linspace(-1, +1, 100)
dBoundary = (-intercept -(parameterWeight0*uniformX1Values)) / parameterWeight1
# plot everything 
plt.title('Scatter plot of actual vs predicted values when C=1')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.plot(uniformX1Values, dBoundary, 'k--', label = 'Decision boundary')
plt.legend(title = 'legend')
plt.show()

# for 100
plt.scatter(X1[y==1], X2[y==1], color = 'green', marker = '+', label = 'Actual +1', s = 100)
plt.scatter(X1[y==-1], X2[y==-1], color = 'blue', marker = 'o', label = 'Actual -1', s = 100)
# get the y predictions based on the model and map the x values based on the 
# y predictions using x markers so that they are visible 
y_pred = SVM100model.predict(X)
plt.scatter(X1[y_pred==1], X2[y_pred==1], color = 'red', marker = 'x', label = 'Predicted +1', s = 100)
plt.scatter(X1[y_pred==-1], X2[y_pred==-1], color = 'orange', marker = 'x', label = 'Predicted -1', s = 100)
# get the intercept and the parameter weights and map the boundary based on it 
intercept = SVM100model.intercept_[0]
parameterWeight0, parameterWeight1 = SVM100model.coef_[0]
uniformX1Values = np.linspace(-1, +1, 100)
dBoundary = (-intercept -(parameterWeight0*uniformX1Values)) / parameterWeight1
# plot everything 
plt.title('Scatter plot of actual vs predicted values when C=100')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.plot(uniformX1Values, dBoundary, 'k--', label = 'Decision boundary')
plt.legend(title = 'legend')
plt.show()

#(c)(i)
# add the two X values squared to the regression
XWithSquared = np.column_stack((X1,X2,pow(X1,2),pow(X2,2)))
XSquaredmodel = LogisticRegression(penalty=None, solver='lbfgs')
XSquaredmodel.fit(XWithSquared, y)
print('Intercept for the 4 X value model:', XSquaredmodel.intercept_)
print('Coefficient for the 4 X value model:', XSquaredmodel.coef_)

#(ii)
plt.scatter(X1[y==1], X2[y==1], color = 'green', marker = '+', label = 'Actual +1', s = 100)
plt.scatter(X1[y==-1], X2[y==-1], color = 'blue', marker = 'o', label = 'Actual -1', s = 100)
# get the y predictions based on the model and map the x values based on the 
# y predictions using x markers so that they are visible 
y_pred = XSquaredmodel.predict(XWithSquared)
plt.scatter(X1[y_pred==1], X2[y_pred==1], color = 'red', marker = 'x', label = 'Predicted +1', s = 100)
plt.scatter(X1[y_pred==-1], X2[y_pred==-1], color = 'orange', marker = 'x', label = 'Predicted -1', s = 100)
plt.title('Scatter plot of actual vs predicted values for model with features squared')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.legend(title = 'legend')
plt.show()

#(iii)
numberMinusOnes = sum(y==-1)
numberPlusOnes = sum(y==+1)
if numberPlusOnes > numberMinusOnes:
    mostCommonClass = +1
else:
    mostCommonClass = -1

baselinePrediction = sum(y==mostCommonClass) / len(y)
baselinePredictionPercent = baselinePrediction*100
print('The baseline prediction is', baselinePredictionPercent, '%')
y_pred = XSquaredmodel.predict(XWithSquared)
XSquaredPrediction = sum(y==y_pred) / len(y)
XSquaredPredictionPercent = XSquaredPrediction*100
print('The X squares model prediction is', XSquaredPredictionPercent, '%')

#(iv)
plt.scatter(X1[y==1], X2[y==1], color = 'green', marker = '+', label = 'Actual +1', s = 100)
plt.scatter(X1[y==-1], X2[y==-1], color = 'blue', marker = 'o', label = 'Actual -1', s = 100)
# get the y predictions based on the model and map the x values based on the 
# y predictions using x markers so that they are visible 
y_pred = XSquaredmodel.predict(XWithSquared)
plt.scatter(X1[y_pred==1], X2[y_pred==1], color = 'red', marker = 'x', label = 'Predicted +1', s = 100)
plt.scatter(X1[y_pred==-1], X2[y_pred==-1], color = 'orange', marker = 'x', label = 'Predicted -1', s = 100)
XValues = np.linspace(-1, +1, 100)
squaredIntercept = XSquaredmodel.intercept_[0]
firstParameter, secondParameter, thirdParameter, fourthParameter = XSquaredmodel.coef_[0]
# aX2^2 + bX2 + C
# a = fourthParameter
# b = secondParameter
# c = firstParameter(X1) + fourthParamter(X^2) + intercept
# the formula for X2 is X2 = -b +/- square root of (b^2 - 4ac) / 2a
a = fourthParameter
b = secondParameter
c = (firstParameter * XValues) + (thirdParameter * (pow(XValues,2))) + squaredIntercept
#X2Part1 = ((-b - np.sqrt(pow(b,2) - (4*a*c))) / (2*a)) this one is the wrong curve
X2Part2 = ((-b - np.sqrt(pow(b,2) - (4*a*c))) / (2*a))
plt.title('Scatter plot of actual vs predicted values for model with features squared with decision boundary')
plt.xlabel('X_1')
plt.ylabel('X_2')
#plt.plot(XValues, X2Part2, 'k--', label = 'Decision boundary') this one is the wrong curve
plt.plot(XValues, X2Part2, 'k--', label = 'Decision boundary')
plt.legend(title = 'legend')
plt.show()

















