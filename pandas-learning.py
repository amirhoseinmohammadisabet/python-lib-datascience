import pandas as pd
import numpy as np

arr = [0,1,2,3]
s1 = pd.Series(arr)

n = np.random.random(4)
s2 = pd.Series(n, index=['a', 'b', 'c', 'd'] )

#create series from dictionary
d = {'a':1, 'b':2, 'c':3, 'd':4}
s3 = pd.Series(d)

s1.index = ['a', 'b', 'c', 'd']
#s4=s1.append(s2)

s5=s2.add(s3)
#max-min-median-add-sub-mul-dev

#create dataframe 1
dates = pd.date_range('today', periods=6)
num_arr = np.random.rand(6,4)
columns = ['A','B','C','D']
df1 = pd.DataFrame(num_arr, index=dates, columns= columns)
print(df1)

#create dataframe 1
data = {'animal': ['cat', 'dog', 'snake', 'bat', 'hen', 'beatle', 'panda'],
        'age': [5, 9, 8, 9, 2, 3, 10],
        'visits': [1, 5 ,2, 3 ,3, 8, 9],
        'priority': ['yes','no','yes','no','yes','no','yes']
        }
lables = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
df2 = pd.DataFrame(data, index= lables)
print(df2.head(4))
#print(df2.tail(2))
#print(df2.values)
#print(df2.index)
#print(df2.columns)


