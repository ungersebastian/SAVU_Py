# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:58:58 2019

@author: ungersebastian
"""

import pandas as pd

#from spc import spc


class LabeledFrame(pd.DataFrame):
    _metadata = ['custom_attr','reserved_attr']
    
    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return label(*args, **kwargs).__finalize__(self)
        return _c
    
    def __init__(self, *args, **kwargs):
        
        super(spc, self).__init__(args, kwargs, is_label = True)
    
    def __repr__(self):
        return pd.DataFrame.__repr__(self)