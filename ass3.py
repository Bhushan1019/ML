"""
Every year many students give the GRE exam to get admission in foreign Universities. The data 
set contains GRE Scores (out of 340), TOEFL Scores (out of 120), University Rating (out of 5), 
Statement of Purpose strength (out of 5), Letter of Recommendation strength (out of 5), 
Undergraduate GPA (out of 10), Research Experience (0=no, 1=yes), Admitted (0=no, 1=yes). 
Admitted is the target variable. 
Data Set Available on kaggle (The last column of the dataset needs to be changed to 0 or 1)
Data Set: https://www.kaggle.com/mohansacharya/graduate-admissions 
The counselor of the firm is supposed check whether the student will get an admission or not 
based on his/her GRE score and Academic Score. So to help the counselor to take appropriate 
decisions build a machine learning model classifier using Decision tree to predict whether a 
student will get admission or not. 
A. Apply Data pre-processing (Label Encoding, Data Transformation….) techniques if 
necessary. 
B. Perform data-preparation (Train-Test Split) 
C. Apply Machine Learning Algorithm 
D. Evaluate Model.
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier 
# Import train_test_split function
from sklearn.model_selection import train_test_split 
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 


data = pd.read_csv("")
data.head()


data.info()


data.describe()


data.isnull().sum()


print("There are {} rows and {} columns".format(data.shape[0],data.shape[1]))


data = data.drop(['Serial No.'],axis=1)
data


data.info()


data['Classlabel'].value_counts()


data['SOP'].value_counts()


data(data.Classlabel==1)


plt.figure(figsize=(10,20))
sns.countplot(data['SOP'].values)


#split dataset in features and target variable
feature_cols = ['GRE Score','TOEFL Score','University Rating','SOP','CGPA','Research']
X = data[feature_cols] # Features
y = data.Classlabel # Target variable


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


len(X_train)


len(X_test)


# Create Decision Tree classifer object
dtclf = DecisionTreeClassifier()

# Fit Decision Tree Classifer
dtclf = dtclf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = dtclf.predict(X_test)
print(y_pred)


# Model Accuracy, how often is the classifier correct?
accuracy=metrics.accuracy_score(y_pred,y_test)*100
print("Accuracy of the model is {:.2f}".format(accuracy))


metrics.confusion_matrix(y_pred,y_test)


pip install graphviz


pip install pydotplus


from sklearn import tree
tree.plot_tree(dtclf)
