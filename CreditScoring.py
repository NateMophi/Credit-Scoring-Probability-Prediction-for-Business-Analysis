import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

dataset = pd.read_excel("a_Dataset_CreditScoring.xlsx")
print(dataset.head())
dataset = dataset.drop("ID", axis=1) #drop entire column that has ID
print(dataset.head())

dataset.isna().sum() #checks for missing vals
dataset = dataset.fillna(dataset.mean()) 

y = dataset.iloc[:,0].values
X = dataset.iloc[:,1:28].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Model Performance
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

predictions = classifier.predict_proba(X_test)
print(predictions)

df_prediction_prob = pd.DataFrame(predictions, columns = ['prob_0', 'prob_1'])
df_prediction_target = pd.DataFrame(classifier.predict(X_test), columns=['predicted_TARGET'])
df_test_dataset = pd.DataFrame(y_test, columns=['Actual Outcome'])

dfx = pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)
dfx.to_excel("c1_Model_Prediction.xlsx")
print(dfx.head())