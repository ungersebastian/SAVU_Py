# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:33:21 2019

@author: ungersebastian
"""

import numpy as np
import pandas as pd

class MISlice(object):
    def __init__(self, *args):
        self.slice_matrix = list(range(len(args)))
        for i_arg, arg in enumerate(args):
            if isinstance(arg, list) or isinstance(arg, tuple):
                self.slice_matrix[i_arg] = np.unique(np.concatenate([np.r_[x] for x in arg]))
            else:
                self.slice_matrix[i_arg] = np.sort(np.r_[arg])
        self.slice_matrix = pd.MultiIndex.from_product(self.slice_matrix)
    

        
        
