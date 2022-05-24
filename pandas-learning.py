import inline as inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

arr = [0, 1, 2, 3]
s1 = pd.Series(arr)

n = np.random.random(4)
s2 = pd.Series(n, index=['a', 'b', 'c', 'd'])

# create series from dictionary
d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
s3 = pd.Series(d)

s1.index = ['a', 'b', 'c', 'd']
# s4=s1.append(s2)

s5 = s2.add(s3)
# max-min-median-add-sub-mul-dev
# print(s1)


# create dataframe 1
dates = pd.date_range('today', periods=6)
num_arr = np.random.rand(6, 4)
columns = ['A', 'B', 'C', 'D']
df1 = pd.DataFrame(num_arr, index=dates, columns=columns)
# print(df1)

# create dataframe 1
data = {'animal': ['cat', 'dog', 'snake', 'bat', 'hen', 'beatle', 'panda'],
        'age': [5, 9, 8, 9, np.nan, 3, 10],
        'visits': [1, 5, np.nan, 3, 3, 8, 9],
        'priority': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes']
        }
lables = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
df2 = pd.DataFrame(data, index=lables)
# print(df2.head(4))
# print(df2.tail(2))
# print(df2.values)
# print(df2.index)
# print(df2.columns)


# print(df2.describe())  #see statistical data of dataframe
# print(df2.T)
# print(df2.sort_values(by='age')[0:5]) #sort all by age and shows just the first five
# print(df2[['age', 'visits']]) #just shows two of columns
# print(df2[['age']].mean())

string = pd.Series(['A', 'B', 'C', 'MMS', 'Ready', np.nan, 'CRipTo'])
# print(string.str.upper())
# print(string.str.lower())

# opperation for Dataframe missing value
df4 = df2.copy()
mean_age1 = df2[['age']].mean()
# print(mean_age1)
mean_age2 = df2[['age']].fillna(5).mean()  # replacing NaN with some int
# print(mean_age2)

# dataframe file operation

df4.to_csv('animal.csv')
#df4.to_excel('animal.xlsx', sheet_name='animal1')
#df6 = pd.read_excel('animal.xlsx', 'animal1', index_col=None, na_values=['NA'])
#df5 = pd.read_csv('animal.csv')
#print(df5)

#series and dataframe line chart
ts = pd.Series(np.random.rand(50), index=pd.date_range('today', periods=50))
ts.plot()
plt.show()