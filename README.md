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

#### create a linear regression model 
regressor = LinearRegression() 
regressor.fit(X_train, y_train) 


#### save the model 
filename = 'linear_model.sav'
pickle.dump(regressor, open(filename, 'wb')) 

#### load the model 
load_model = pickle.load(open(filename, 'rb')) 

y_pred = load_model.predict(X_test) 
print('root mean squared error : ', np.sqrt( 
	metrics.mean_squared_error(y_test, y_pred))) 


## 2 Problem 2 : Data Preprocessing, Analysis, and Visualization for building a Machine learning model		(https://www.geeksforgeeks.org/data-preprocessing-analysis-and-visualization-for-building-a-machine-learning-model/)
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 

dataset = pd.read_csv('Churn_Modelling.csv')
dataset.isnull().any() 
dataset["Geography"].fillna(dataset["Geography"].mode()[0],inplace = True) 
dataset["Gender"].fillna(dataset["Gender"].mode()[0],inplace = True) 
dataset["Age"].fillna(dataset["Age"].mean(),inplace = True)
dataset.isnull().any()
le = LabelEncoder() 
dataset['Geography'] = le.fit_transform(dataset["Geography"]) 
dataset['Gender'] = le.fit_transform(dataset["Gender"]) 
x = dataset.iloc[:,3:13].values 
y = dataset.iloc[:,13:14].values
x_train, x_test, y_train, y_test = train_test_split(x,y, 
													test_size = 0.2, 
													random_state = 0)
sc = StandardScaler() 
x_train = sc.fit_transform(x_train) 
x_test = sc.fit_transform(x_test)
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 

from sklearn import metrics 

knn = KNeighborsClassifier(n_neighbors=3) 
rfc = RandomForestClassifier(n_estimators = 7, criterion = 'entropy', random_state =7) 
svc = SVC() 
lc = LogisticRegression() 

#### making predictions on the training set 
for clf in (rfc, knn, svc,lc): 
	clf.fit(x_train, y_train) 
	y_pred = clf.predict(x_test) 
	print("Accuracy score of ",clf.__class__.__name__,"=", 
		100*metrics.accuracy_score(y_test, y_pred))


## 3 Problem 3 : Building a Machine Learning Model Using J48 Classifier	(https://www.geeksforgeeks.org/building-a-machine-learning-model-using-j48-classifier/)
// Java Program for Creating a Model Based on J48 Classifier

// Importing required classes
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

// Main class
public class BreastCancer {

	// Main driver method
	public static void main(String args[])
	{

		// Try block to check for exceptions
		try {

			// Creating J48 classifier
			J48 j48Classifier = new J48();

			// Dataset path
			String breastCancerDataset
				= "/home/droid/Tools/weka-3-8-5/data/breast-cancer.arff";

			// Create bufferedreader to read the dataset
			BufferedReader bufferedReader
				= new BufferedReader(
					new FileReader(breastCancerDataset));

			// Create dataset instances
			Instances datasetInstances
				= new Instances(bufferedReader);

			// Set Target Class
			datasetInstances.setClassIndex(
				datasetInstances.numAttributes() - 1);

			// Evaluation
			Evaluation evaluation
				= new Evaluation(datasetInstances);

			// Cross Validate Model with 10 folds
			evaluation.crossValidateModel(
				j48Classifier, datasetInstances, 10,
				new Random(1));
			System.out.println(evaluation.toSummaryString(
				"\nResults", false));
		}

		// Catch block to check for rexceptions
		catch (Exception e) {

			// Print and display the display message
			// using getMessage() method
			System.out.println("Error Occurred!!!! \n"
							+ e.getMessage());
		}

		// Display message to be printed ion console
		// when program is successfully executed
		System.out.print("Successfully executed.");
	}
}

## 4 Problem 4 : How to Get Regression Model Summary from Scikit-Learn	(https://www.geeksforgeeks.org/how-to-get-regression-model-summary-from-scikit-learn/)
#### Import packages 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 

#### Load the data 
irisData = load_iris() 

#### Create feature and target arrays 
X = irisData.data 
y = irisData.target 

#### Split data into train and test sets 
X_train, X_test, y_train, y_test = train_test_split( 
	X, y, test_size=0.2, random_state=42) 

model = LinearRegression() 

model.fit(X_train, y_train) 

#### predicting on the X_test data set 
print(model.predict(X_test)) 

#### summary of the model 
print('model intercept :', model.intercept_) 
print('model coefficients : ', model.coef_) 
print('Model score : ', model.score(X, y)) 



