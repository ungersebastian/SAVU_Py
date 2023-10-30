# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:03:51 2023

@author: basti
"""

import numpy as np
from scipy.stats import poisson

n_images = 2**7
n_batch = 2**3
n_batch_images = n_images//n_batch
n_bit = 16
b_max = np.sum(2**np.arange(n_bit))
b_min = 0

v1 = []
v2 = []
r = []

for n_iter in range(b_max+1):
    

    dist = poisson.rvs(mu=n_iter, size=n_images).astype(np.uint16)
    
    a1 = np.sum(dist//n_images)
    a2 = np.sum(np.array([np.sum(d//n_batch_images) for d in np.reshape(dist, (n_batch, n_batch_images))])//n_batch)
    
    v1.append(a1)
    v2.append(a2)
    r.append(n_iter)
    
    print( n_iter, a1, a2 )
    
#%%

import matplotlib.pyplot as plt

m = 2000

fig, ax = plt.subplots()
ax.plot(r[:m], v1[:m], label='Solution 2')
ax.plot(r[:m], v2[:m], label='Solution 3')
ax.plot(r[:m], r[:m], label='Mean')
legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.show()

plt.figure()
plt.plot(r[:m], v1[:m])
plt.plot(r[:m], v2[:m])
plt.plot(r[:m], r[:m])


plt.figure()
plt.plot(r[m:2*m], v1[m:2*m])
plt.plot(r[m:2*m], v2[m:2*m])
plt.plot(r[m:2*m], r[m:2*m])
