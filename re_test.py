# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:21:09 2019

@author: ungersebastian
"""

class get_attribute(object):
    def __init__(self, parent):
        super(get_attribute, self).__init__()
        self.parent = parent

    def __getattribute__(self, name):
        if name == 'parent':
            return object.__getattribute__(self,name)
        else:
            return object.__getattribute__(self.parent,name)
    
    def __getitem__(self, name):
        return self.__getattribute__(name)

class tester(object):
    def __init__(self):
        self.a1 = 2
        self.ga = get_attribute(parent = self)
    

    
        
a = tester()
print(a.a1)
print(a.ga.a1)
print(a.ga['a1'])

