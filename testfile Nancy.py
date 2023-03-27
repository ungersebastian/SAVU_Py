# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:30:12 2019

@author: ungersebastian
"""

import savu_py as sap
import numpy as np
import matplotlib.pyplot as plt
#import os

from savu_py.read import witec as witec

from pkg_resources import resource_filename, resource_listdir

# Itemize data files under proj/resources/images:
#path = resource_filename("SpectralAnalysisPack","resources")

#path = 'D:/Promotion/Raman Daten/Export/ChlamydiaAbortus/In_BGM-Zellen/36h_pi';
path = 'N:/Daten_Promotions_Sebastian/20200226'
#%%

names = [
    '20200226 Coinfection_Chlam_Cox_54h_z-stack_004_Spec.Data 1_F (Header).txt',
    '20200226 Coinfection_Chlam_Cox_54h_z-stack_005_Spec.Data 1_F (Header).txt',
    '20200226 Coinfection_Chlam_Cox_54h_z-stack_006_Spec.Data 1_F (Header).txt',
    '20200226 Coinfection_Chlam_Cox_54h_z-stack_007_Spec.Data 1_F (Header).txt',
    '20200226 Coinfection_Chlam_Cox_54h_z-stack_008_Spec.Data 1_F (Header).txt',
    '20200226 Coinfection_Chlam_Cox_54h_z-stack_009_Spec.Data 1_F (Header).txt'
    ]
res = []

for n in names:
    
    head = witec.from_Witec_head(n, path, kind = 'scan');
    
    v = head.values[:,:-1]
    nChan = np.sqrt(v.shape[0]).astype(int)
    v = np.reshape(v, (nChan, nChan, v.shape[1])).astype(float)
    i = np.sum(v, axis = -1)
    res.append(v)
    plt.figure()
    plt.imshow(i)
    plt.show()

res = np.array(res)
np.save(r'N:\Daten_Promotions_Sebastian\raman3D', res)
#%%



x = witec.import_spc(head[0], head[1])

print(x[1])
#if 'GraphName' in head :
 #   print(head['GraphName'])
#xValues = xValues.split(('\n')[0])


#fig = plt.figure()
#plt.plot(x[0],x[1][0,0])
#plt.show()

#test = x[1]



#plt.imshow(x, cmap='hot', interpolation='nearest')
# plt.show()
