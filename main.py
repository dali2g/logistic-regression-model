import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

dataset = pd.read_csv('diabetes.csv')
dataset.head(6)

dataset.tail(6)
dataset.shape
dataset.info
dataset.hist(figsize=(20, 20))
dataset['Outcome'].value_counts()
x = dataset[
    ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = dataset['Outcome']

Scaler = StandardScaler()
print(Scaler.fit_transform(x))

xx = Scaler.fit_transform(x)
print(xx)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x_train, x_test, y_train, y_test = train_test_split(xx, y, test_size=0.2)

classifier = LogisticRegression()
classifier.fit(x_train, y_train)
train_prediction = classifier.predict(x_train)
accuracy = accuracy_score(train_prediction, y_train)
print(accuracy)

y_perd = classifier.predict(x_test)
accuracy = accuracy_score(y_perd, y_test)
print(accuracy)

from sklearn import metrics
import seaborn as sns

cnf_matrix = metrics.confusion_matrix(y_test, y_perd)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
print(p)

input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = Scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
if prediction[0] == 0:
    print("La personne n'est pas diabétique")
else:
    print("La personne est diabétique")
