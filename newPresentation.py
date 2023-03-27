# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:30:12 2019

@author: ungersebastian
"""
#%%

import savu_py as savu
import numpy as np
import matplotlib.pyplot as plt
from pkg_resources import resource_filename


savu.spc.OPTIONS['n_col_example_pm'] = 10
savu.spc.OPTIONS['n_col_example_max'] = 3

savu.spc.OPTIONS['n_row_example_pm'] = 4
savu.spc.OPTIONS['n_row_example_max'] = 10

shape = (50,75,15)
y_mat_1 = np.random.rand(*shape)
my_spc = savu.spc(y_mat_1, unit_wl = 'kHz', is_spc = True)
print(my_spc)
my_label = savu.spc(y_mat_1, unit_wl = 'kHz', is_spc = False)
print(my_label)

#%%

# k-means cluster analysis
cluster = savu.cluster.kmeans(my_spc, n_clusters = 3)

#%% 
# it is also possible to initialize an empty savu object and fill it later

my_empty_one = savu.spc()
print(my_empty_one)
my_empty_one.spc = y_mat_1
print(my_empty_one)

#%%

# k-means cluster analysis
cluster = savu.cluster.kmeans(my_spc.spc.values, n_clusters = 3)
#%%
#predict something
c_test = cluster.ga.predict(my_spc) # by default, spc or first label will be used
c_train = cluster.cluster.values

print(c_train.T[0]-c_test)

# get only cluster and scores
c1 = savu.cluster.kmeans(my_spc.spc.values, n_clusters = 3, returns = ['scores', 'cluster', 'predict'])
# get only predictor
c2 = savu.cluster.kmeans(my_spc.spc.values, n_clusters = 3, returns = 'predict')
test_2 = c2(my_label)


#%%
#examples

# attributes
n_rows = my_spc.ga.n_rows
print(n_rows)
# setting an attribute
key, val = ('k1', 12)
my_spc.ga(key, val)
my_spc.ga.k2 = 23
print(my_spc)

#%%
# labels

# setting an label
tf = np.random.rand(n_rows)
my_spc['TrueFalse'] = tf

tf[tf>0.5]=True
tf[tf!=True]=False

my_spc.tf = tf.astype(bool)
g = np.arange(np.prod(shape[:-1]))
my_spc.pos = g
g = np.reshape(g, shape[:-1])
print(my_spc)


#%%

# getting object

# using labels
small_spc_1 = my_spc[my_spc.TrueFalse>0.5] # direct comparisson of values without using .values
print(small_spc_1)

#%%
# image slicing
small_spc_2 = my_spc[1:6, 2:4]
g2 = g[1:6, 2:4]
f2 = small_spc_2.pos.values
print(small_spc_2)
#%%
# label indexing
small_spc_3 = my_spc['spc',slice(1,5)]
print(small_spc_3)

# wavelength indexing
small_spc_4 = my_spc['wl',5]
print(small_spc_4)
small_spc_5 = my_spc['wl',2:5, 0, [0,1,2,3],0.2, 0.7]
print(small_spc_5)
#%%

# deleting stuff

del_spc = my_spc.__copy__()
del_spc.ga.wavelength = del_spc.ga.wavelength * 2

print(del_spc.ga.wavelength)

print(my_spc.ga.wavelength)

#%%



# Itemize data files under proj/resources/images:
path = resource_filename("SpectralAnalysisPack","resources")
fname = '20171123 DC59_36h_Cell2_011_Spec.Data 1_F (Header).txt'

my_spc_real = savu.read.from_Witec_head(fname, path)



