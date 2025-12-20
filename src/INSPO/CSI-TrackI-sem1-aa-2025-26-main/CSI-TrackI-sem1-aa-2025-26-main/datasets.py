import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder

def calculate_params(matrix): 
    params = []
    for i in range(matrix.shape[1] - 1):
        mean = np.mean(matrix[:,i])
        std = np.std(matrix[:,i])
        perc25 = np.percentile(matrix[:,i],25)
        perc50 = np.percentile(matrix[:,i],50)
        perc75 = np.percentile(matrix[:,i],75)
        f = np.fft.fft(matrix[:,i])
        spEnergy = np.sum(f**2)/len(f)
        params.append(mean)
        params.append(std)
        params.append(perc25)
        params.append(perc50)
        params.append(perc75)
        params.append(spEnergy)
    params.append(matrix[0,3])
    return params

def import_accelerometer():
    dataframes = []
    for i in range(15):
        string = "dataset/Activity Recognition from Single Chest-Mounted Accelerometer/" + str(i+1) + ".csv"
        dataframes.append(pd.read_csv(string,header=None))
        dataframes[i].columns = ["pos","x","y","z","action"]

    for i,dataframe in enumerate(dataframes):
        dataframes[i] = dataframe.drop("pos",axis=1)

    
    dataset = np.zeros((1,19))
    for dataframe in dataframes:
        matrix = dataframe.to_numpy()
        matrices = []
        matrices.append(matrix[matrix[:,3]==1])
        matrices.append(matrix[matrix[:,3]==2])
        matrices.append(matrix[matrix[:,3]==3])
        matrices.append(matrix[matrix[:,3]==4])
        matrices.append(matrix[matrix[:,3]==5])
        matrices.append(matrix[matrix[:,3]==6])
        matrices.append(matrix[matrix[:,3]==7])
        for i,matrix in enumerate(matrices):
            if i != 1:
                stop = False
                index = 0
                while(not stop):
                    if index + 1000 > matrix.shape[0]:
                        sample = matrix[index:matrix.shape[0]]
                        if sample.shape[0] != 0:
                            features = calculate_params(sample)
                        stop = True
                    else:
                        sample = matrix[index:index + 1000]
                        features = calculate_params(sample)
                    index += 1000
                    dataset = np.insert(dataset,dataset.shape[0],features,axis=0)
            else:
                sample = matrix
                features = calculate_params(sample)
                dataset = np.insert(dataset,dataset.shape[0],features,axis=0)

    dataset = np.delete(dataset,0,0)
    np.random.seed(0) #for reproducibility
    np.random.shuffle(dataset)
    n_samples = dataset.shape[0]
    X_train = dataset[:int(0.7*n_samples),:18]
    Y_train = dataset[:int(0.7*n_samples),18] - 1
    X_val = dataset[int(0.7*n_samples):int(0.8*n_samples),:18]
    Y_val = dataset[int(0.7*n_samples):int(0.8*n_samples),18] - 1
    X_test = dataset[int(0.8*n_samples):,:18]
    Y_test = dataset[int(0.8*n_samples):,18] - 1
    mean = np.mean(X_train,axis=0)
    std = np.std(X_train,axis=0)
    X_train = (X_train - mean)/std
    X_val = (X_val - mean)/std
    X_test = (X_test - mean)/std

    return X_train, X_val, X_test, Y_train, Y_val, Y_test
