## Scikit1

### 1 Problem 1 : Save and Load Machine Learning Models in Python with scikit-learn	(https://www.geeksforgeeks.org/save-and-load-machine-learning-models-in-python-with-scikit-learn/)
#### import packages 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn import metrics 
import pickle 

##### import the dataset 
dataset = pd.read_csv('headbrain1.csv') 

X = dataset.iloc[:, : -1].values 
Y = dataset.iloc[:, -1].values 

##### train test split 
X_train, X_test, y_train, y_test = train_test_split( 
	X, Y, test_size=0.2, random_state=0) 

# create a linear regression model 
regressor = LinearRegression() 
regressor.fit(X_train, y_train) 


# save the model 
filename = 'linear_model.sav'
pickle.dump(regressor, open(filename, 'wb')) 

# load the model 
load_model = pickle.load(open(filename, 'rb')) 

y_pred = load_model.predict(X_test) 
print('root mean squared error : ', np.sqrt( 
	metrics.mean_squared_error(y_test, y_pred))) 


### 2 Problem 2 : Data Preprocessing, Analysis, and Visualization for building a Machine learning model		(https://www.geeksforgeeks.org/data-preprocessing-analysis-and-visualization-for-building-a-machine-learning-model/)

### 3 Problem 3 : Building a Machine Learning Model Using J48 Classifier		(https://www.geeksforgeeks.org/building-a-machine-learning-model-using-j48-classifier/)

### 4 Problem 4 : How to Get Regression Model Summary from Scikit-Learn	(https://www.geeksforgeeks.org/how-to-get-regression-model-summary-from-scikit-learn/)
