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
print(np.unique(spc_info[:,0]))

#%%

import savu_py as sa

my_spc = sa.spc(spc_in, unit_wl = 'nm')
my_spc.date = spc_info[:,0]
my_spc.type = spc_info[:,1]

#spc_test = my_spc[(my_spc.date == 20171124) ^ (my_spc.date == 20171125)]
spc_test = my_spc[(my_spc.date == 20171125) ]
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

disp.plot()
plt.show()

print(classification_report(y_true, y_pred))

#%% FIT on LDA -- 3!

from scipy.optimize import leastsq

fitfunc  = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)
errfunc  = lambda p, x, y: (y - fitfunc(p, x))

hist_rb, bin_edges_rb = np.histogram(transform[spc_test.type.values.flatten() == 'RB'], bins = n_bins, density=True)
x_rb = (bin_edges_rb[1:]+bin_edges_rb[:-1])/2

init  = [1.0, 0.5, 0.5]
out_rb   = leastsq( errfunc, init, args=(x_rb, hist_rb))[0]

multifit = lambda p, x: ( 
    p[0]*np.exp(-0.5*((x-p[1])/p[2])**2) +
    p[3]*np.exp(-0.5*((x-p[4])/p[5])**2) +
    p[6]*np.exp(-0.5*((x-p[7])/p[8])**2)
    )
multierr = lambda p, x, y: (y - multifit(p, x))
multinit = [
    0.1,-5.0,0.5,
    0.2,-2.0,0.5,
    0.7,-0.5,0.5
    ]

hist_eb, bin_edges_eb = np.histogram(transform[spc_test.type.values.flatten() == 'EB'], bins = n_bins, density=True)
x_eb = (bin_edges_eb[1:]+bin_edges_eb[:-1])/2

out_eb   = leastsq( multierr, multinit, args=(x_eb, hist_eb))[0]

plt.figure()
plt.hist(transform[spc_test.type.values.flatten() == 'RB'], label='RB', bins=n_bins, alpha=.7, edgecolor='red', density = True)
plt.hist(transform[spc_test.type.values.flatten() == 'EB'], label='EB', bins=n_bins, alpha=.7, edgecolor='green', density = True)
plt.legend()
plt.plot(x_eb, multifit(out_eb, x_eb))
plt.plot(x_rb, fitfunc(out_rb, x_rb))
plt.plot(x_eb, fitfunc(out_eb[0:3], x_eb))
plt.plot(x_eb, fitfunc(out_eb[3:6], x_eb))
plt.plot(x_eb, fitfunc(out_eb[6:9], x_eb))
plt.show()

#%% binning erschaffen

spc_test.lda_score = transform

## binning: RB
bin_rb = np.full(transform.shape, -1)
for ix, (xu, xo) in enumerate(zip(bin_edges_rb[:-1], bin_edges_rb[1:])):
    bin_rb[(xu <= transform) * (xo > transform)] = ix
## binning: EB
bin_eb = np.full(transform.shape, -1)
for ix, (xu, xo) in enumerate(zip(bin_edges_eb[:-1], bin_edges_eb[1:])):
    bin_eb[(xu <= transform) * (xo > transform)] = ix
    
spc_test.bin_rb = bin_rb
spc_test.bin_eb = bin_eb

#%% wichtung RB

classes = np.sort(np.unique(spc_test.bin_rb))[1:]

res_rb = np.sum([ np.mean(spc_test[spc_test.bin_rb == c].spc) * fitfunc(out_rb, x_rb[c]) for c in classes], axis = 0)
weights_rb = np.sum([ fitfunc(out_rb, x_rb[c]) for c in classes])

res_rb = res_rb / weights_rb

#%% wichtung EB_1-3

classes = np.sort(np.unique(spc_test.bin_eb))[1:]
res_eb_1 = np.sum([ np.mean(spc_test[spc_test.bin_eb == c].spc) * fitfunc(out_eb[0:3], x_eb[c]) for c in classes], axis = 0)
weights_eb_1 = np.sum([ fitfunc(out_eb[0:3], x_eb[c]) for c in classes])
res_eb_1 = res_eb_1 / weights_eb_1

res_eb_2 = np.sum([ np.mean(spc_test[spc_test.bin_eb == c].spc) * fitfunc(out_eb[3:6], x_eb[c]) for c in classes], axis = 0)
weights_eb_2 = np.sum([ fitfunc(out_eb[3:6], x_eb[c]) for c in classes])
res_eb_2 = res_eb_2 / weights_eb_2

res_eb_3 = np.sum([ np.mean(spc_test[spc_test.bin_eb == c].spc) * fitfunc(out_eb[6:9], x_eb[c]) for c in classes], axis = 0)
weights_eb_3 = np.sum([ fitfunc(out_eb[6:9], x_eb[c]) for c in classes])
res_eb_3 = res_eb_3 / weights_eb_3


#%% Summennormierung

res_rb_norm = res_rb/np.sum(res_rb)
res_eb_1_norm = res_eb_1/np.sum(res_eb_1)
res_eb_2_norm = res_eb_2/np.sum(res_eb_2)
res_eb_3_norm = res_eb_3/np.sum(res_eb_3)

# Plot:
plt.figure()
plt.plot(res_rb_norm, label='RB')
plt.plot(res_eb_1_norm, label='EB_1')
plt.plot(res_eb_2_norm, label='EB_2')
plt.plot(res_eb_3_norm, label='EB_3')
plt.legend()
plt.show()
    

#%% FIT on LDA -- 3+2!
# Ergebnis: Keine Eindeutige Aufspaltung bei RB

from scipy.optimize import leastsq

fitfunc  = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)

multifit_rb = lambda p, x: ( 
    p[0]*np.exp(-0.5*((x-p[1])/p[2])**2) +
    p[3]*np.exp(-0.5*((x-p[4])/p[5])**2)
    )
multierr = lambda p, x, y: (y - multifit_rb(p, x))

hist_rb, bin_edges_rb = np.histogram(transform[spc_test.type.values.flatten() == 'RB'], bins = n_bins, density=True)
x_rb = (bin_edges_rb[1:]+bin_edges_rb[:-1])/2

multinit = [
    0.2,-2.0,0.5,
    0.7,0.5,0.5
    ]
out_rb   = leastsq( multierr, multinit, args=(x_rb, hist_rb))[0]

multifit = lambda p, x: ( 
    p[0]*np.exp(-0.5*((x-p[1])/p[2])**2) +
    p[3]*np.exp(-0.5*((x-p[4])/p[5])**2) +
    p[6]*np.exp(-0.5*((x-p[7])/p[8])**2)
    )
multierr = lambda p, x, y: (y - multifit(p, x))
multinit = [
    0.1,-5.0,0.5,
    0.2,-2.0,0.5,
    0.7,-0.5,0.5
    ]

hist_eb, bin_edges_eb = np.histogram(transform[spc_test.type.values.flatten() == 'EB'], bins = n_bins, density=True)
x_eb = (bin_edges_eb[1:]+bin_edges_eb[:-1])/2

out_eb   = leastsq( multierr, multinit, args=(x_eb, hist_eb))[0]

plt.figure()
plt.hist(transform[spc_test.type.values.flatten() == 'RB'], label='RB', bins=n_bins, alpha=.7, edgecolor='red', density = True)
plt.hist(transform[spc_test.type.values.flatten() == 'EB'], label='EB', bins=n_bins, alpha=.7, edgecolor='green', density = True)
plt.legend()
plt.plot(x_eb, multifit(out_eb, x_eb))
plt.plot(x_rb, multifit_rb(out_rb, x_rb))
plt.plot(x_eb, fitfunc(out_eb[0:3], x_eb))
plt.plot(x_eb, fitfunc(out_eb[3:6], x_eb))
plt.plot(x_eb, fitfunc(out_eb[6:9], x_eb))
plt.plot(x_rb, fitfunc(out_rb[0:3], x_rb))
plt.plot(x_rb, fitfunc(out_rb[3:6], x_rb))
plt.show()
