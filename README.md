# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm

 # 1.START
 # 2.Import the required libraries.
 # 3.Upload the csv file and read the dataset.
 # 4.Check for any null values using the isnull() function.
 # 5.From sklearn.tree inport DecisionTreeRegressor.
 # 6.Import metrics and calculate the Mean squared error.
 # 7.Apply metrics to the dataset, and predict the output.
 # 8.END

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: JANARTHANAN K
RegisterNumber: 212223040072

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
print(data.head())
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

*/
```

## Output:
![Screenshot 2024-10-16 220601](https://github.com/user-attachments/assets/bced0165-3dea-4b53-98ba-40e68f656f86)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
