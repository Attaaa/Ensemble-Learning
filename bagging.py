# Ensemble with bagging method using naive bayes
# created by Muhammad Hatta Eka Putra

import numpy as np
import pandas as pd
import collections
import operator
import naive_bayes as nb

np.random.seed(1)
# import data train and data test
dataTrain = pd.read_csv('TrainsetTugas4ML.csv').values
dataTest = pd.read_csv('TestsetTugas4ML.csv').values
new_dataTest = []
for i in range(len(dataTest)):
    new_dataTest.append(dataTest[i][:2])
dataTest = np.array(new_dataTest)

num_bootstrap = 5
list_bootstrap = {}
len_bootstrap = 100

# create bootstrap with random data from data train
for i in range(num_bootstrap):
    bootstrap = []
    for j in range(len_bootstrap):
        randomData = dataTrain[np.random.randint(1, len(dataTrain))]
        bootstrap.append( randomData )
    list_bootstrap[i] = np.copy(bootstrap)

# predict class result from data test with each bootstrap
model_result = {}
for i in range(num_bootstrap):
    dataTrain = list_bootstrap[i]
    model_result[i] = nb.naiveBayes(dataTrain, dataTest, class_index=2)

print(model_result)
