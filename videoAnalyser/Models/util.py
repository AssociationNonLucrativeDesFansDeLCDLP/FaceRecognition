import numpy as np
import csv
import matplotlib.pyplot as plt

def plot(lists, names, title, p):
    for i in range(len(lists)):
        #if(p[i]['depth']=='2'):
        plt.plot(lists[i], label=names[i])
        plt.legend()
    plt.title(title)
    plt.show()


def strToFloat(values):
    results=[]
    for val in values:
        a=val[1:len(val)-1]
        #print(a)
        b=a.split(', ')
        #print(accuracy)
        c=[float(x) for x in b]
        #print(acc)
        results.append(c)
    return results

def extractValues(file):
    accuracies=[]
    val_accuracies=[]
    loss=[]
    val_loss=[]
    names=[]
    params=[]
    with open(file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            names.append(row['name'])
            accuracies.append(row['accuracy'])
            val_accuracies.append(row['val_accuracy'])
            loss.append(row['loss'])
            val_loss.append(row['val_loss'])
            p=dict()
            p['n_dense']=row['n_dense']
            p['depth']=row['depth']
            p['epochs']=row['n_epochs']
            p['eval']=row['eval']
            #p['fold']=row['fold']
            params.append(p)
    accuracies=strToFloat(accuracies)
    val_accuracies=strToFloat(val_accuracies)
    loss=strToFloat(loss)
    val_loss=strToFloat(val_loss)
    return names, accuracies, val_accuracies, loss, val_loss, params
