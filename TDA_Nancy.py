# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:30:12 2019

@author: ungersebastian
"""
#%%

import savu_py as savu
import numpy as np
import matplotlib.pyplot as plt


savu.spc.OPTIONS['n_col_example_pm'] = 10
savu.spc.OPTIONS['n_col_example_max'] = 3

savu.spc.OPTIONS['n_row_example_pm'] = 4
savu.spc.OPTIONS['n_row_example_max'] = 10



# Itemize data files under proj/resources/images:
path = r'N:\TDA_Nancy'
fname = '20171122 DC59_36h_Cell1_012_Spec.Data 1_F 1 (Header).txt'

my_spc_real = savu.read.from_Witec_head(fname, path)

#%%
with open(path + '/' + fname, 'r') as f:
    f_rl = f.readlines()

print(f_rl)
#%%



