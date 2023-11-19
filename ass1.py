# Perform following operation on given dataset. 
# a) Find Shape of Data 
# b) Find Missing Values 
# c) Find data type of each column 
# d) Finding out Zero's 
# e) Find Mean age of patients 
# f) Now extract only Age, Sex, ChestPain, RestBP, Chol. Randomly divide dataset in training 
#  (75%) and testing (25%). 
# Through the diagnosis test I predicted 100 report as COVID positive, but only 45 of those were actually 
# positive. Total 50 people in my sample were actually COVID positive. I have total 500 samples. 
# Create confusion matrix based on above data and find 
# I Accuracy 
# II Precision 
# III Recall 
# IV F-1 score

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import hvplot.pandas
from scipy import stats

%matplotlib inline
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


data = pd.read_csv("")
data.head()


data.info()


data.shape
data.size


data.describe()


data.notnull()


data.values


data["Age"].mean()


data.isin([0]).any()
(data==0).sum()


from sklearn.model_selection import train_test_split

X=data[['ChestPain', 'Age', 'Sex', 'RestBP', 'Chol']]
y = data[['RestBP', 'Chol']]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=10)
X_train