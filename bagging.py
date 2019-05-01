# import data training
# dari data training di random dan dibuat boostrap dengan random index dari data train
# test data test dengan tiap data boostrap yang sudah di dapat dengan naive bayes
# voting data yang di dapat dari naive bayes
# Ensemble with bagging method using naive bayes
# created by Muhammad Hatta Eka Putra

import numpy as np
import pandas as pd
import collections
import operator
import naive_bayes as nb
import matplotlib.pyplot as plt

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

# voting all model for final result
final_result = []
for i in range(len(dataTest)):
    temp = [ model_result[model_idx][i] for model_idx in model_result ]
    temp = collections.Counter(temp)
    result = max(temp.items(), key=operator.itemgetter(1))[0]
    final_result.append(result)
print(final_result)

# code above is aptional
# plotting final result
for i in range(len(dataTest)):
    if (final_result[i] == 1):
        plt.scatter(dataTest[i][0], dataTest[i][1], c="blue")
    else:
        plt.scatter(dataTest[i][0], dataTest[i][1], c="red")

# plotting data Train
# new_dataTest = []
# _class = []
# for i in range(len(dataTrain)):
#     _class.append(dataTrain[i][2])
#     new_dataTest.append(dataTrain[i][:2])
# dataTrain = np.array(new_dataTest)

# for i in range(len(dataTrain)):
#     if (_class[i] == 1):
#         plt.scatter(dataTrain[i][0], dataTrain[i][1], c="green")
#     else:
#         plt.scatter(dataTrain[i][0], dataTrain[i][1], c="black")

plt.show()
