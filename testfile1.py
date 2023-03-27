# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:30:12 2019

@author: ungersebastian
"""

import SpectralAnalysisPack as sap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce
from operator import mul
# simulate spectra

sap.MIFrame.OPTIONS['n_col_example_pm'] = 1
sap.MIFrame.OPTIONS['n_col_example_max'] = 3

sap.MIFrame.OPTIONS['n_row_example_pm'] = 3
sap.MIFrame.OPTIONS['n_row_example_max'] = 10



nw = 200
nx = 50
n_class = 5

wavelength = np.arange(nw)/nw *500 + 500

def gaus1d(x, amp = 700, sig = 1, x0 = 750):
    return amp * np.exp(- (x - x0)**2/(2*sig**2))
def random(i_min, i_max):
    return np.random.rand(1) * (i_max-i_min) + i_min


shape = (50,10,5)
y_mat_1 = np.random.rand(*shape)

a = sap.MIFrame(y_mat_1)
#%%

shape = (50,5,5,20)
y_mat_2 = np.random.rand(*shape)
shape = (50,5,5,1)
y_mat_3 = np.random.rand(*shape)

name = 'my_label'
name_list = np.arange(shape[-1])

a = sap.MIFrame(y_mat_1, y_mat_2, testlabel = y_mat_3)
a.ga.my_attr_1 = 4
a.ga.my_attr_2 = [1,23,4]
a.ga.my_attr_3 = np.arange(42)
b = sap.MIFrame(y_mat_1, y_mat_2, testlabel = y_mat_3)
c = b.unnamed_1
del b.testlabel
#b=a.testlabel


#%%

a = sap.spc(y_mat_1)

#%%
a = sap.MIFrame(y_mat_1, y_mat_2, testlabel = y_mat_3)

a.ga.my_attr_1 = 4
a.ga.my_attr_2 = [1,23,4]
a.ga.my_attr_3 = np.arange(42)

b = sap.MISlice(
        ([5,6], range(2)),
        [4,2],
        (3,2)
        )

a[b]

#%%


#import os

from pkg_resources import resource_filename, resource_listdir

# Itemize data files under proj/resources/images:
path = resource_filename("SpectralAnalysisPack","resources")

#path = 'D:/Promotion/Raman Daten/Export/ChlamydiaAbortus/In_BGM-Zellen/36h_pi';

head = sap.read.header_Witec('20171123 DC59_36h_Cell2_011_Spec.Data 1_F (Header).txt', path, kind = 'scan');
x = sap.read.import_spc(head[0], head[1])


fig = plt.figure()
plt.plot(x[0],x[1][0,0])
plt.show()

im = np.sum(x[1], axis = 2)
mean = np.mean(im)
sd = np.std(im)

im[im>mean+5*sd] = np.min(im)

plt.imshow(im , cmap='hot', interpolation='nearest')
plt.show()