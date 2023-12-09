# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:51:56 2023

@author: basti
"""


### imports of standard libs
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #### NEED version 2.2.3
import os



### Import of training data
path_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

filename = join(path_dir, 'Topological-Data-Analysis', 'TopologicalDataAnalysis', 'resources', 'spc_export.csv')

def read_csv(filename, chunksize = 10**5):
    
    chunks = pd.read_csv(
        filename,
        chunksize = chunksize,
        header = None,
        sep=',',
        escapechar='\\'
        )
    
    df = pd.concat(chunks)
    
    return df.values

spc_in = read_csv(filename)

filename = join(path_dir, 'Topological-Data-Analysis', 'TopologicalDataAnalysis', 'resources', 'spc_export_info.csv')

spc_info = read_csv(filename)


#%%

def pairs(data, *args, **kwargs):
    if data.ndim == 2:
        nComp = len(data)
    elif data.ndim == 1:
        nComp = 1
    else:
        print('Warning: data.ndim not valid')
        return None
    if 'stretch_factor' in kwargs:
        stretch_factor = kwargs['stretch_factor']
    else:
        stretch_factor = 0.1
    
    if nComp > 1:
        pRange = np.array([[np.quantile(d,0.01), np.quantile(d,0.99)] for d in data])
        d = (pRange[:,1]-pRange[:,0])*stretch_factor
        pRange[:,0] -= d
        pRange[:,1] += d
        fig, axs = plt.subplots(nComp, nComp)
        
        for i1 in np.arange(0,nComp):
            axs[i1,i1].hist(data[i1],bins = 20, range=(pRange[i1,0],pRange[i1,1]))
            axs[i1,i1].set_xlim(pRange[i1,0],pRange[i1,1])
            axs[i1,i1].set_xlabel(''.join(['PC - ', str(i1+1)]))
            for i2 in np.arange(i1+1, nComp):
                axs[i1,i2].scatter(data[i2], data[i1], alpha=0.7, s = 1)
                axs[i1,i2].set_xlim(pRange[i2,0],pRange[i2,1])
                axs[i1,i2].set_ylim(pRange[i1,0],pRange[i1,1])
                axs[i1,i2].set_xlabel(''.join(['PC - ', str(i2+1)]))
                axs[i1,i2].set_ylabel(''.join(['PC - ', str(i1+1)]))
                axs[i2,i1].scatter(data[i1], data[i2], alpha=0.7, s = 1)
                axs[i2,i1].set_xlim(pRange[i1,0],pRange[i1,1])
                axs[i2,i1].set_ylim(pRange[i2,0],pRange[i2,1])
                axs[i2,i1].set_xlabel(''.join(['PC - ', str(i1+1)]))
                axs[i2,i1].set_ylabel(''.join(['PC - ', str(i2+1)]))
        
        
        plt.show()

#%%

import savu_py as sa

my_spc = sa.spc(spc_in, unit_wl = 'nm')
my_spc.date = spc_info[:,0]
my_spc.type = spc_info[:,1]

spc_test = my_spc[(my_spc.date == 20171124) ^ (my_spc.date == 20171125)]

#%% PCA

from sklearn.decomposition import PCA

n_components = 4

pca = PCA(n_components=n_components)
pca.fit(spc_test.spc.values)
print('Explained variance: ', pca.explained_variance_ratio_)

scores = pca.transform(spc_test.spc.values).astype(float)
loadings = pca.components_

plt.figure()
for l in loadings:
    plt.plot(l)
plt.show()

label = spc_test.type.values.flatten()
color = ['red' if l == 'RB' else 'green' for l in label]
plt.figure()
plt.scatter(scores[:,0], scores[:,1], c = color)
plt.show()
    
#%% LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

lda = LinearDiscriminantAnalysis()
lda.fit(spc_test.spc.values, spc_test.type.values.flatten())

predict = lda.predict(spc_test.spc.values)    
transform = lda.transform(spc_test.spc.values)    

n_bins = 100

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(transform[spc_test.type.values.flatten() == 'RB'], bins=n_bins, density = True)
axs[1].hist(transform[spc_test.type.values.flatten() == 'EB'], bins=n_bins,  density = True)

plt.figure()
plt.hist(transform[spc_test.type.values.flatten() == 'RB'], label='RB', bins=n_bins, alpha=.7, edgecolor='red', density = True)
plt.hist(transform[spc_test.type.values.flatten() == 'EB'], label='EB', bins=n_bins, alpha=.7, edgecolor='green', density = True)
plt.legend()
plt.show()

loadings = lda.scalings_

plt.figure()
plt.plot(loadings)
plt.show()

y_true = spc_test.type.values.flatten()
y_pred = predict
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lda.classes_)

plt.figure()
disp.plot()
plt.show()

print(classification_report(y_true, y_pred))

#%% PCA-LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

n_components = 10
threshold = 0.01

pca = PCA(n_components=n_components)
pca.fit(spc_test.spc.values)
evr =  pca.explained_variance_ratio_
print('Explained variance: ', evr)
n_comp = np.sum(evr > threshold)
print('n_comp used: ', n_comp)
scores = pca.transform(spc_test.spc.values).astype(float)[:,:n_comp]
loadings = pca.components_[:n_comp]

lda = LinearDiscriminantAnalysis()
lda.fit(scores, spc_test.type.values.flatten())

predict = lda.predict(scores)    
transform = lda.transform(scores)    

n_bins = 100

plt.figure()
plt.hist(transform[spc_test.type.values.flatten() == 'RB'], label='RB', bins=n_bins, alpha=.7, edgecolor='red', density = True)
plt.hist(transform[spc_test.type.values.flatten() == 'EB'], label='EB', bins=n_bins, alpha=.7, edgecolor='green', density = True)
plt.legend()
plt.show()

loading = np.sum( lda.scalings_ * loadings, axis = 0)
plt.figure()
plt.plot(loading)
plt.show()

y_true = spc_test.type.values.flatten()
y_pred = predict
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lda.classes_)

plt.figure()
disp.plot()
plt.show()

print(classification_report(y_true, y_pred))