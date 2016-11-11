#Importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Providing required Style
style.use('ggplpot')

#Upload the data for analysis
train = pd.read_csv('train_.csv')
train = pd.DataFrame(train)

test = pd.read_csv('test_.csv')
test = pd.DataFrame(test)


#checking missing values
print(train.apply(lambda x:sum(x.isnull())))

#Filling missing values
value_to_impute = ['Workclass','Occupation','Native.Country']
for v in value_to_impute:
    train[v].fillna('-99999',inplace=True)
    test[v].fillna('-99999',inplace=True)


#check outlinear
train.plot('ID','Hours.Per.Week',kind='scatter')

#variable transformation
var_data_type = list(train.dtypes.loc[train.dtypes == 'object'].index)
print(train[var_data_type].apply(lambda x: len(x.unique())))


#Modify the data for better performance
# for i in var_data_type:
#     frq = train[i].value_counts()/train.shape[0]
#     category = frq.loc[frq.values <0.05].index
#     for j in category:
#         train[i].replace({j:'Others'},inplace=True)
#         if j == 'Income.Group':
#             continue
#         test[i].replace({j: 'Others'}, inplace=True)


#Change the data types of column
le = LabelEncoder()
var_data_type1 = train.dtypes.loc[train.dtypes == 'object'].index
for i in var_data_type1:
    train[i]=le.fit_transform(train[i])
    if i == 'Income.Group':
        continue
    test[i] = le.fit_transform(test[i])

dependent_variable = 'Income.Group'
Independent_variable = [x for x in train.columns if x not in ['ID',dependent_variable]]



#Prediction from given data
model = DecisionTreeClassifier(max_depth=10,min_samples_leaf=100,max_features='sqrt')
model.fit(train[Independent_variable],train[dependent_variable])
predict = model.predict(train[Independent_variable])
predict_test = model.predict(test[Independent_variable])

#Analyse accuracy of Data
acc_train = accuracy_score(train[dependent_variable],predict)
print('Train accuracy : %f'%acc_train)

#Converting data into specific format

df = pd.DataFrame()
df['ID'] = test['ID']
df['Income.Group'] = predict_test
df['Income.Group'] = df['Income.Group'].replace(0,'<=50K')
df['Income.Group'] = df['Income.Group'].replace(1,'>50K')

#converting data into csv form
df.to_csv('Solution.csv',index=False)
