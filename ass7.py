"""
Download the dataset of National Institute of Diabetes and Digestive and Kidney 
Diseases from below link : 
https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indiansdiabetes.data.csv 
The dataset is has total 9 attributes where the last attribute is “Class attribute” having 
values 0 and 1. (1=”Positive for Diabetes”, 0=”Negative”) 
a. Load the dataset in the program. Define the ANN Model with Keras. Define 
 at least two hidden layers. Specify the ReLU function as activation function 
 for the hidden layer and Sigmoid for the output layer. 
b. Compile the model with necessary parameters. Set the number of epochs and 
 batch size and fit the model. 
c. Evaluate the performance of the model for different values of epochs and 
 batch sizes. 
d. Evaluate model performance using different activation functions Visualize 
 the model using ANN Visualizer.
 """


import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError


df = pd.read_csv("")
df


df.shape


x = df[:,:8]
y = df[:,8]


from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


print(f"x train shape{X_train.shape}")
print(f"y train shape{y_train.shape}")
print(f"x test shape{X_test.shape}")
print(f"y test shape{y_test.shape}")
print(f"x val shape{X_val.shape}")
print(f"y val shape{y_val.shape}")


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)


from collections import Counter
Counter(y)


import seaborn as sns


from tensorflow.keras.models import Sequential


model = Sequential([

tf.keras.layers.InputLayer(8,),
Dense(50,activation='relu'),
    
Dense(50,activation='relu'),
Dense(50,activation='relu'),
Dense(50,activation='relu'),
    
Dense(1,activation='sigmoid')
])


model.summary()


opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


history = model.fit(x=x,y=y,epochs=300, batch_size=50,validation_data=(X_val,y_val))


losses = pd.DataFrame(model.history.history)
losses.plot()


model.evaluate(x,y)


y_pred = model.predict(X_test)


y_pred


# !pip install ann_visualizer


# !pip install graphviz


# !pip3 install keras
# !pip3 install ann_visualizer
# !pip install graphviz


# from ann_visualizer.visualize import ann_viz;

# ann_viz(model, title="My first neural network")



# python3 index.py