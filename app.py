import pandas as pd
import numpy
import scipy
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Load training data - CHANGE LOCATION AS NECESSARY
file_name = r'c:/data/hackerrank_email/training_dataset.csv'
data = pd.read_csv(file_name)
print(data.shape)

#Create subset dataframe with most important columns
data_sub = data[['clicked', 'unsubscribed', 'submissions_count_master_7_days', 'contest_login_count_30_days', 'contest_login_count_7_days', 'hacker_confirmation','opened']]

#Convert boolean fields to binary
data_sub['opened'] = data_sub['opened'].astype(int)
data_sub['clicked'] = data_sub['clicked'].astype(int)
data_sub['unsubscribed'] = data_sub['unsubscribed'].astype(int)
data_sub['hacker_confirmation'] = data_sub['hacker_confirmation'].astype(int)

#Create X and Y from subset
X = data_sub.values[:,0:6]
Y = data_sub.values[:,6]

#Create training set
test_size = 0.33
train_test_seed = 7
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=train_test_seed)

#Create model
model = LogisticRegression()
model.fit(X_train, Y_train)


