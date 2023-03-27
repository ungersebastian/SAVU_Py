# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:28:45 2021

@author: basti
"""

import numpy as np
from sklearn.decomposition import PCA
from ..spc import spc
from ..utils.__predict__ import __predict__
from ..utils.norm import norm

def pca(data = None, n_comp = None, returns = ['loadings','scores','predict'], *args, **kwargs):
    if isinstance(data, spc):
        index = data.index
        wl = data.ga.wavelength
        data = data.__get_first_label__()
        is_spc = True
    else:
        wl = np.arange(data.shape[-1])
        is_spc = False
    
    if 'center' in kwargs:
        center = kwargs.pop('center')
    else:
        center = True
    if 'norm' in kwargs:
        pnorm = kwargs.pop('norm')
    else:
        pnorm = False
    if 'axis' in kwargs:
        axis = kwargs.pop('axis')
    else:
        axis = 1
        
    if pnorm != False:
        data = norm(data, norm = pnorm, apply = True, axis = axis, *args, **kwargs)
    if center == True:
        center = np.mean(data, axis = axis-1)
        data = data-center
    
    pca_obj = PCA(n_comp, *args, **kwargs)
    pca_fit = pca_obj.fit(X = data)
    
    if not isinstance(returns,(list, tuple, np.ndarray)):
        returns = [returns]
    
    #generate empty spc and atr object
    atr = []
    obj = spc(is_spc = False) 
    
    # 1 - items
    if ('all' in returns) or ('scores' in returns):
        obj['scores'] = pca_fit.transform(data)
    
    # 2 - attributes
    if ('all' in returns) or ('loadings' in returns):
        loadings = spc(spc = pca_obj.components_, wavelength = wl)
        atr.append(['loadings',loadings])
        
    if ('all' in returns) or ('predict' in returns):
        name = ''.join([
            'PCA predict, n_comp = ', str(n_comp)        
        ])  
        predict = __predict__(pca_fit.transform, name) 
        atr.append(['predict',predict])
        
    if ('all' in returns) or ('fit' in returns):
        atr.append(['PCA fit',pca_fit])
        
    if ('all' in returns) or ('pca' in returns):
        atr.append(['PCA object',pca_obj])
    
    if ('all' in returns) or ('explained_variance' in returns):
        atr.append(['explained_variance',pca_obj.explained_variance_])
    
    if ('all' in returns) or ('explained_variance_ratio' in returns):
        atr.append(['explained_variance_ratio',pca_obj.explained_variance_ratio_])
    
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
