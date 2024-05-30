import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the dataset to a Pandas DataFrame
df = pd.read_csv('/content/credit_data.csv')

# first 5 rows of the dataset
df.head()

# To find the number of missing values
df.isnull().sum()

# distribution of legit transactions & fraudulent transactions
df['Class'].value_counts()

# pie chart to visualize fraud and non-fraud transactions
fraud = len(df[df['Class']==1])
notfraud = len(df[df['Class']==0])

labels = 'Fraud','Not Fraud'
sizes = [fraud,notfraud]

plt.figure(figsize=(10,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=0)
plt.title('Ration of Fraud vs Not-Fraud\n', fontsize=20)
sns.set_context("paper",font_scale=1.8)

# separating the data for analysis
legit = df[df.Class == 0]
fraud = df[df.Class == 1]

# downsampling the legit transactions
legit_sample = legit.sample(n=492)

# create a new dataframe by concating the downsampled data
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# spliting the data into targets
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# split the data into testing and training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X.shape, X_train.shape, X_test.shape)

# use logistic regression to classifier
model = LogisticRegression()
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)
