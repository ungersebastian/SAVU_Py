# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 10:08:29 2021

@author: basti
"""

import numpy as np
from ..spc import spc

class __predict__(object):
    def __init__(self, fun = None, name = None):
        if isinstance(fun, type(None)) or isinstance(name, type(None)):
            raise TypeError('Function and name must be given')
            return None
        self.fun = fun
        self.name = str(name)
        super(__predict__,self).__init__()
        
    def __call__(self, data):
        if isinstance(data, spc):
            return self.fun(data.__get_first_label__())
        else:
            return self.fun(data)
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name

