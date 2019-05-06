# Ensemble with bagging method using naive bayes
# created by Muhammad Hatta Eka Putra

import numpy as np
import pandas as pd
import collections
import operator
import csv
import naive_bayes as nb

np.random.seed(1)
# import data train and data test
dataTrain = pd.read_csv('Trainset.csv').values
dataTest = pd.read_csv('Testset.csv').values
new_dataTest = []
for i in range(len(dataTest)):
    new_dataTest.append(dataTest[i][: 2])
dataTest = np.array(new_dataTest)

num_bootstrap = 5
len_bootstrap = 100

# create bootstrap with random data from data train
list_bootstrap = {}
for i in range(num_bootstrap):
    bootstrap = []
    for j in range(len_bootstrap):
        randomData = dataTrain[np.random.randint(0, len(dataTrain))]
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

# save result in csv file
data_file = open( 'Result.csv', 'w', newline = '' )
with data_file:
    writer = csv.writer(data_file)
    for row in final_result:
        writer.writerow([row])
