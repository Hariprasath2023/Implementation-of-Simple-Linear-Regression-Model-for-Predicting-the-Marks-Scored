# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*

Program to implement the linear regression using gradient descent.
Developed by: hari prasath R.K
RegisterNumber:212223040055


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
Dataset:

![Screenshot 2024-03-06 102327](https://github.com/Hariprasath2023/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145207783/c6196c72-58bf-4d5e-9e24-f7ea553cf9e5)

Head Values

![Screenshot 2024-03-06 102431](https://github.com/Hariprasath2023/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145207783/14a6c0ac-d54f-4765-859b-31025a1e2b88)

Tail Values

![Screenshot 2024-03-06 102438](https://github.com/Hariprasath2023/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145207783/852d5413-9c26-4358-b7f8-0ea5234eedc1)

X and Y values
![Screenshot 2024-03-06 102500](https://github.com/Hariprasath2023/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145207783/42308ab1-1606-49ef-9092-3a689eab7b0e)

Predication values of X and Y
![Screenshot 2024-03-06 102509](https://github.com/Hariprasath2023/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145207783/558be3c7-7c40-4d8d-a205-422ad468570f)

MSE,MAE and RMSE:

![Screenshot 2024-03-06 102517](https://github.com/Hariprasath2023/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145207783/26b025b6-0667-4ced-b634-a82179f899f9)

Training Set

![Screenshot 2024-03-06 102543](https://github.com/Hariprasath2023/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145207783/9dae1369-d75f-4907-a113-a8ab763bb29c)

Testing Set

![Screenshot 2024-03-06 102554](https://github.com/Hariprasath2023/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145207783/b11beb87-967e-4af1-ae4d-db703400323c)










## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
