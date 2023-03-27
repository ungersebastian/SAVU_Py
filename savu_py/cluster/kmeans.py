# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:46:08 2021

@author: basti
"""

import numpy as np
from sklearn.cluster import KMeans
from ..spc import spc
from ..utils.__predict__ import __predict__

def kmeans(data = None, n_clusters = None, returns = ['cluster','center','predict'], *args, **kwargs):
    if isinstance(data, spc):
        index = data.index
        wl = data.ga.wavelength
        data = data.__get_first_label__()
        is_spc = True
    else:
        wl = np.arange(data.shape[-1])
        is_spc = False
        
    kmeans_obj = KMeans(n_clusters = n_clusters)
    kmeans_fit = kmeans_obj.fit(X = data)
    
    if not isinstance(returns,(list, tuple, np.ndarray)):
        returns = [returns]
    
    #generate empty spc and atr object
    atr = []
    obj = spc(is_spc = False) 
    
    # 1 - items
    if ('all' in returns) or ('cluster' in returns):
        obj['cluster'] = kmeans_fit.predict(data)
    if ('all' in returns) or ('scores' in returns):
        obj['scores'] = kmeans_fit.transform(data)
    
    # 2 - attributes
    if ('all' in returns) or ('centers' in returns):
        center = spc(spc = kmeans_fit.cluster_centers_, wavelength = wl)
        atr.append(['center',center])
        
    if ('all' in returns) or ('predict' in returns):
        name = ''.join([
            'kMeans predict, n_clusters = ', str(n_clusters)        
        ])  
        predict = __predict__(kmeans_fit.predict, name) 
        atr.append(['predict',predict])
        
    if ('all' in returns) or ('transform' in returns):
        name = ''.join([
            'kMeans transform, n_clusters = ', str(n_clusters)        
        ])  
        transform = __predict__(kmeans_fit.transform, name) 
        atr.append(['transform',transform])
        
    if ('all' in returns) or ('fit' in returns):
        atr.append(['kMeans fit',kmeans_fit])
    
    
    
    if is_spc:
        obj.index = index
        
    if obj.ga.is_empty == False:
        for a in atr:
            obj.ga(a[0], a[1])    
    else:
        if len(atr) > 1:
            obj = {}
            for a in atr:
                obj.__setitem__(a[0],a[1])
        else:
            obj = atr[0][1]
    return obj
