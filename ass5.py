# Download the following customer dataset from below link: Data Set: 
# https://www.kaggle.com/shwetabh123/mall-customers 
# This dataset gives the data of Income and money spent by the customers visiting a Shopping Mall. The data 
# set contains Customer ID, Gender, Age, Annual Income, and Spending Score. Therefore, as a mall owner you 
# need to find the group of people who are the profitable customers for the mall owner. Apply at least two 
# clustering algorithms (based on Spending Score) to find the group of customers. 
# a. Apply Data pre-processing (Label Encoding , Data Transformation….) techniques if necessary. 
# b. Perform data-preparation( Train-Test Split) 
# c. Apply Machine Learning Algorithm 
# d. Evaluate Model. 
# e. Apply Cross-Validation and Evaluate Model 


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')


df = pd.read_csv("")
df.head()


df.tail()


df.shape


df.columns


df.drop("CustomerID",axis=1,inplace=True)
df


print("Missing values: ")
df.isnull().sum()


df.describe()


df.info()


df.nunique()


df.corr()


plt.figure(figsize=(7,5))
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn')
plt.show()


df.hist(bins = 50,figsize = (10,6))


# pie chart for "Total Gender Count"
df['Genre'].value_counts().plot(kind='pie',figsize=(5,5),autopct='%1.1f%%') 
plt.title("Total Gender Count") 
plt.show() 


sns.pairplot(df,hue="Genre"); 


sns.set(style = 'whitegrid') 
sns.scatterplot(y = 'Spending Score (1-100)',x ='Annual Income (k$)',data = df,hue= "Genre"); 
plt.title('Iris Dataset') 
plt.show() 


# LabelEncoder for encoding binary categories in a column
from sklearn.preprocessing import LabelEncoder 
from sklearn import metrics 
le = LabelEncoder() 
# One single vector so it is ovbious what we want to encode
df["Genre"] = le.fit_transform(df["Genre"])


df


# Finding the optimum number of clusters using k-means
data = df.copy() 
x = data.iloc[:,[2,3]] 
#importing Kmean model
from sklearn.cluster import KMeans 
wcss = [] 
for i in range(1,11): 
 kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42) 
 kmeans.fit(x) 
 # appending the WCSS to the list 
 #(kmeans.inertia_ returns the WCSS value for an initialized cluster)
 wcss.append(kmeans.inertia_) 
 print('k:',i ,"-> wcss:",kmeans.inertia_)


# Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1,11),wcss,marker='o') 
plt.title('The Elbow Method') 
plt.xlabel('Number of Clusters') 
plt.ylabel('WCSS') 
plt.show()


#Taking 5 clusters
km1=KMeans(n_clusters=5) 
#Fitting the input data
km1.fit(data) 
#predicting the labels of the input data
y=km1.predict(data) 
#adding the labels to a column named label
data["label"] = y 
#The new dataframe with the clustering done
data.head() 


#Scatterplot of the clusters
plt.figure(figsize=(6,4)) 
sns.scatterplot(x = 'Annual Income (k$)',y = 'Spending Score (1-100)',hue="label", 
 palette=['green','brown','orange','red','dodgerblue'],data = data ) 
plt.xlabel('Annual Income (k$)') 
plt.ylabel('Spending Score (1-100)') 
plt.title('Spending Score (1-100) vs Annual Income (k$)') 
plt.show()


X=data.iloc[:,:4] 
y=data.iloc[:,-1] 


# Splitting of dataset into train and test
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Shape of train Test Split
print(X_train.shape,y_train.shape) 
print(X_test.shape,y_test.shape) 


from sklearn.cluster import KMeans 
km=KMeans(n_clusters=5) 
km.fit(X_train) 
#predicting the target value from the model for the samples
y_train_km = km.predict(X_train) 
y_test_km = km.predict(X_test) 


from sklearn.metrics.cluster import adjusted_rand_score 
acc_train_gmm = adjusted_rand_score(y_train,y_train_km) 
acc_test_gmm = adjusted_rand_score(y_test,y_test_km) 
print("K mean : Accuracy on training Data: {:.3f}".format(acc_train_gmm)) 
print("K mean : Accuracy on test Data: {:.3f}".format(acc_test_gmm))


data = df.copy() 
data = data.iloc[:,[2,3]] 
data 


sns.scatterplot(x="Annual Income (k$)",y="Spending Score (1-100)",data = data )


import scipy.cluster.hierarchy as shc 
dendrogram = shc.dendrogram(shc.linkage(data,method="ward")) 
plt.title("dendrogram Plot") 
plt.xlabel("Customer") 
plt.ylabel("Eclidean Distance") 
plt.grid(False) 


from sklearn.cluster import AgglomerativeClustering 
agc = AgglomerativeClustering(n_clusters=5) 
data["label"] = agc.fit_predict(data) 
data 


#Scatterplot of the clusters
sns.scatterplot(x = 'Annual Income (k$)',y = 'Spending Score (1-100)',hue="label", 
 palette=['green','brown','orange','red','dodgerblue'],data = data ) 
plt.xlabel('Annual Income (k$)') 
plt.ylabel('Spending Score (1-100)') 
plt.title('Spending Score (1-100) vs Annual Income (k$)') 
plt.show() 
