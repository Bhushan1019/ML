# This data consists of temperatures of INDIA averaging the temperatures of all places month 
# wise. Temperatures values are recorded in CELSIUS 
# a. Apply Linear Regression using suitable library function and predict the Month-wise 
# temperature. 
# b. Assess the performance of regression models using MSE, MAE and R-Square metrics 
# c. Visualize simple regression model. 


import numpy as np      # for Numeric Operations 
import pandas as pd     # For Dataframe Operations 
import matplotlib.pyplot as plt     # For Plotting and Visualization 
from sklearn.linear_model import LinearRegression       #sklearn implementation of LinearRegress 
from sklearn import linear_model,metrics

trainData = pd_read_csv("")
trainData.head(n=10)


trainData.dtypes
trainData.columns


trainData.describe()


trainData.isnull().sum()


# top_10_data = trainData.nlargest(10,"ANNUAL")
# plt.figure(figsize=(14,12))
# plt.title("Top 10 temperature records")
# sns.barplot(x=top_10_data.YEAR,y=top_10_data.ANNUAL)


X=trainData[["YEAR"]]
y=trainData[["JAN"]]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
len(X_train)


len(X_test)


trainData.shape


reg = linear_model.LinearRegression()
print(X_train)


model = reg.fit(X_train,y_train)


r_sq = reg.score(X_train,y_train)
print("Determination coefficient: ",r_sq)
print('Intercept: ',model.intercept_)
print("Slope: ",model.coef_)


y_pred = model.predict(X_test)
print('predicted response:',Y_pred,sep='\n')


plt.scatter(X_train,y_train,color='black')
plt.plot(X_train,reg.predict(X_train),color='blue',linewidth=3)
plt.title("Temperature vs Year")
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.show()


plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='black',linewidth=3)
plt.title("Temperature vs Year")
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.show()


mse = np.sum((Y_pred - y_test)**2)
rmse = np.sqrt(mse/24)
print("Mean Squared Error(MSE):", mse) 
print("Root Mean Squared Error(RMSE):", rmse)


SSR = np.sum((Y_pred - Y_test)**2)      #Sum of square of Residuals/Errors SSR/SSE 
SST = np.sum((Y_test - np.mean(Y_test))**2)         # total sum of squares 
r2_score = 1 - (SSR/SST)        # R2 score 
print('SST:', SST) 
print('SSR', SSR) 
print('R2 square:', r2_score) 