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

#Load training data
file_name = r'c:/data/hackerrank_email/training_dataset.csv'
data = pd.read_csv(file_name)
print(data.shape)

#Create subset dataframe with most important columns
data_sub = data[['clicked', 'unsubscribed', 'submissions_count_master_7_days', 'contest_login_count_30_days', 'contest_login_count_7_days', 'hacker_confirmation','opened']]

