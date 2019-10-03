import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
#%matplotlib inline 
from sklearn import preprocessing

dfs_train= pd.read_csv('train.csv')

df_test = pd.read_csv('test.csv')
df_train = dfs_train[['Qualification','Gender','Trainings_Attended','Year_of_birth','Last_performance_score','Year_of_recruitment'
              ,'Targets_met','Previous_Award','Training_score_average', 'Promoted_or_Not']]

df_test = df_test[['Qualification','Gender','Trainings_Attended','Year_of_birth','Last_performance_score','Year_of_recruitment'
              ,'Targets_met','Previous_Award','Training_score_average']]
#test data
int_test = df_test.select_dtypes(include=['int64']).copy()
float_test=df_test.select_dtypes(include=['float64']).copy()
df_int_float_test = pd.concat([float_test,int_test], axis=1, join_axes=[int_test.index])
obj_test =  df_test.select_dtypes(include=['object']).copy()

le = LabelEncoder()
obj_test_trf=obj_test.astype(str).apply(le.fit_transform)
df_final_test = pd.concat([df_int_float_test,obj_test_trf], axis=1, join_axes=[df_int_float_test.index])
df_final_test['Qualification'].fillna(0, inplace=True)


#train data
int_df = df_train.select_dtypes(include=['int64']).copy()
float_df=df_train.select_dtypes(include=['float64']).copy()
df_int_float = pd.concat([float_df,int_df], axis=1, join_axes=[int_df.index])
obj_df =  df_train.select_dtypes(include=['object']).copy()


le = LabelEncoder()
obj_df_trf=obj_df.astype(str).apply(le.fit_transform)
df_final = pd.concat([df_int_float,obj_df_trf], axis=1, join_axes=[df_int_float.index])
df_final['Qualification'].fillna(3, inplace=True)

# join both dataset

#df_final['Promoted_or_Not'].fillna(0.0, inplace=True)


X = df_final.drop(['Promoted_or_Not'],1)
X = preprocessing.scale(X)
y = df_final['Promoted_or_Not']


Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.3)

model = MLPClassifier(max_iter=1000,random_state=42)
#model = SVC(, random_state=42, gamma='auto')
model.fit(Xtrain,ytrain)

accuracy = model.score(Xtest,ytest)
print(accuracy)
df_final_test = preprocessing.scale(df_final_test)
measure = model.predict(df_final_test)

EmployeeNo = dfs_train['EmployeeNo'][0:16496]
output=pd.DataFrame({'EmployeeNo':EmployeeNo,'Promoted_or_Not':measure})
print(output['Promoted_or_Not'].value_counts())
filename = 'Submission2.csv'

output.to_csv(filename,index=False)
print('Saved file: ' + filename)