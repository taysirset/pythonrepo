#!/usr/bin/env python
# coding: utf-8

#  Importing the dataset

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys

# variable for filename 
if len(sys.argv) < 2: 
    print ('Missing filename')
    sys.exit(-1)
    
filename = sys.argv[1]

print('loading your filename{}'.format(filename))

#  use read_csv() to read regrex1.csv file

dataset = pd.read_csv(filename)
print(dataset)

# plot data 

plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.title('y vs x for {}'.format(filename))
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('py_orig{}.png'.format(filename))
plt.show()

# ## Fitting linear regression to the Dataset
model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])

# adjusted r-squared

model.score(dataset[['x']], dataset[['y']])

# visualizing Taylor's linear regression results
plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
plt.title('Combined Plot {}'.format(filename))
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('py_lm{}.png'.format(filename))
plt.show()




