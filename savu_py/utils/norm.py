# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:59:35 2021

@author: basti
"""

import numpy as np
from ..spc import spc

def norm(data, norm = None, apply = True, *args, **kwargs):
    
    if 'axis' in kwargs:
        axis = kwargs['axis']
    else:
        axis = 1
    
    data = data.__copy__()
    
    if isinstance(data, spc):
        is_spc = True
        values = data.__get_first_label__()
    else:
        is_spc = False
        values = data
    
    values = np.array(values)
    shape = values.shape[~axis]
    
    if norm == None:
        result = np.ones(shape)
    elif isinstance(norm, str):
        if norm == 'minmax':
            if 'use_quantiles' in kwargs:
                use_quantiles = kwargs['use_quantiles']
                if 'quantile' in kwargs:
                    quantile = kwargs['kwargs']
                else:
                    quantile = 0.05
            elif 'quantile' in kwargs:
                use_quantiles = True
                quantile = kwargs['quantile']
            else:
                use_quantiles = False
            
            if use_quantiles == False:
                result = [(np.amax(v) - np.amin(v)) for v in values]
                if apply == True:
                    values = [((v - np.amin(v))/r) for v, r in zip(values, result)]
            else:
                result = [(np.quantile(v, 1-quantile) - np.quantile(v, quantile)) for v in values]
                if apply == True:
                    values = [((v - np.quantile(v, quantile))/r) for v, r in zip(values, result)]
        else:
            print('Warning: norm not implemented')
            result = np.ones(shape)
    else:    
        if norm == np.infty:
            result = np.amax(np.abs(values), axis = axis)
        elif norm > 0:
            result = (np.sum(np.abs(values)**norm, axis = axis))**(1/norm)
        else:
            print('Warning: norm not implemented')
            result = np.ones(shape)
        if apply == True:
            values = np.array([d/n if n > 0 else np.zeros(len(d)) for d, n in zip(values, result)])
    
    if apply == True:
       if is_spc:
           data[data.__get_first_label__(only_name = True)] = values
           return data
       else:
           return values
    else:
        if is_spc:
            index = data.index
            data = spc(norm = result, is_spc = False)
            data.index = index
            return data
        else:
            return result
    