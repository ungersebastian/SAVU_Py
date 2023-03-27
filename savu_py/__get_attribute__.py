# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:34:35 2019

@author: ungersebastian
"""

import pandas as pd
from copy import deepcopy

class __get_attribute__(object):
    def __init__(self, parent):
        super(__get_attribute__, self).__init__()
        self.parent = parent

    def __getattribute__(self, name):
        if name == 'parent':
            return object.__getattribute__(self,name)
        else:
            return self.parent.__ga__(name)
    
    def __getitem__(self, name):
        if name == 'parent':
            return object.__getattribute__(self,name)
        else:
            return self.parent.__ga__(name)
    
    def __setattr__(self, name, value):
        if name == 'parent':
            object.__setattr__(self,name, value)
        else:
            self.parent.__sa__(name, value)
    
    def __setitem__(self, name, value):
        if name == 'parent':
            object.__setattr__(self,name, value)
        else:
            self.parent.__sa__(name, value)
    
    def __delitem__(self, name):
        if name == 'parent':
            object.__delattr__(self,name)
        else:
            self.parent.__da__(name)
            
    def __hasattr__(self, name):
        if name == 'parent':
            object.__hasattr__(self,name)
        else:
            self.parent.__ha__(name)
    
    def __hasitem__(self, name):
        if name == 'parent':
            object.__hasattr__(self,name)
        else:
            self.parent.__ha__(name)
    
    def __call__(self, name, value=None):
        if isinstance(value, type(None)):
            if name == 'parent':
                return object.__getattribute__(self,name)
            else:
                return self.parent.__ga__(name)
        else:
            if name == 'parent':
                object.__setattr__(self,name, value)
            else:
                self.parent.__sa__(name, value)
    
    def __repr__(self):
        return '<SpectralAnalysisPack.__get_attribute__ helper class>'
    