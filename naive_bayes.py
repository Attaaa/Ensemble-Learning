# gaussian naive bayes
# created by Muhammad Hatta Eka Putra

import numpy as np
import operator

def splitDataByClass(data, class_index):
    # split data by given class_index in list data
    new_data = {}
    for i in range(len(data)):
        class_value = data[i][class_index]
        if (class_value not in new_data):
            new_data[class_value] = []
        new_data[class_value].append(data[i][: class_index])
    for _class in new_data:
        new_data[_class] = np.array(new_data[_class])
    return new_data
    
def calculate_mean_var(data):
    # input for this method is data that has separated by class
    # this method will calculate mean and variance value for every data attribute
    # for every classes
    list_mean_var = {}
    for _class in data.keys():
        list_mean_var[_class] = []
        for i in range(data[_class].shape[1]):
            attr_mean = np.mean(data[_class][:, i])
            attr_var = np.var(data[_class][:, i]) * ( data[_class].shape[0]/(data[_class].shape[0]-1) )
            list_mean_var[_class].append([attr_mean, attr_var])
    return list_mean_var

def calculate_post_prob(list_mean_var, input_data):
    # calculate posterior probabilites of input_data for each class
    phi = 3.14
    num_attr = len(input_data)
    pfc = {}
    for _class in list_mean_var:
        temp = 1
        pfc[_class] = []
        for j in range(num_attr):
            mean = list_mean_var[_class][j][0]
            var = list_mean_var[_class][j][1]
            temp *= ( 1/np.sqrt(2*phi*var) * np.exp( -0.5 * (input_data[j] - mean)**2 / var ) )
        pfc[_class] = temp
    return pfc

def naiveBayes(dataTrain, dataTest, class_index):
    num_dataTrain = len(dataTrain)
    data = splitDataByClass(dataTrain, class_index)
    list_mean_var = calculate_mean_var(data)
    list_prediction = []
    for row in dataTest:
        pfc = calculate_post_prob(list_mean_var, row)

        class_prob = {}
        pcf = {}
        # calculate probabilites of each class
        for _class in data:
            class_prob[_class] = len(data[_class]) / num_dataTrain
            pcf[_class] = 1

        # calculate conditional probabilitest of each class
        total_prob = 0.0
        for _class in data:
            total_prob = total_prob + (pfc[_class] * class_prob[_class])
        for _class in data:
            pcf[_class] = (pfc[_class] * class_prob[_class])/total_prob

        prediction = max(pcf.items(), key=operator.itemgetter(1))[0]
        list_prediction.append(prediction)
    return list_prediction