# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:04:57 2019

@author: ungersebastian
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:07:50 2019

@author: ungersebastian
"""
"""
import numpy as np
import pandas as pd
import copy    
# -*- coding: utf-8 -*-
"""
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:07:50 2019

@author: ungersebastian
"""

import numpy as np
import pandas as pd
import copy    

class spc(pd.DataFrame):
    """
    _metadata = {
            'print_attr': True,
            'print_labels': True
            }
    """

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return spc(*args, **kwargs).__finalize__(self)
        return _c
    
    def __init__(self, *args, **kwargs):
        if isinstance(args[0], (pd.DataFrame, pd.core.internals.BlockManager)):
            super(pd.DataFrame, self).__init__(args[0])
        else:
            
            self._custom_attr = dict()
            self._reserved_attr = {
                    'wavelength': None,
                    'unit': None,
                    'n_spc': None,
                    'shape_spc': None,
                    'n_wl': None,
                    'columns': {'unnamed': np.array([])}
                    }
                        
            i_args = 0
            
            # getting the spc data
            if 'spc' in kwargs:
                spcData = np.array(kwargs.pop('spc', None))
            elif len(args)-i_args > 0:
                spcData = np.array(args[i_args])
                i_args += 1
            else:
                raise ValueError('spc must be given!')
                
            self.shape_spc = np.array(spcData.shape[:-1])
            self.n_spc = np.prod(spcData.shape[:-1])
            """
            # dict for reserved attributes
            # setting the wavelength attribute
            if 'wavelength' in kwargs:
                self.wavelength = np.array(kwargs.pop('wavelength', None))
            elif len(args)-i_args > 0:
                self.wavelength = np.array(args[i_args])
                i_args += 1
            else:
                self.wavelength = np.arange(spcData.shape[-1])
            if spcData.shape[-1] != self.wavelength.shape[0]:
                raise ValueError('Wavelength dimension did not fit spectral dimension.\n Expected %s channels, received %s.' % (self.spc.shape[-1],self.wavelength.shape[0]))
            
            # setting the unit attribute
            if 'unit' in kwargs:
                self.unit = str(kwargs.pop('unit', None))
            elif len(args)-i_args > 0:
                self.unit = str(args[i_args])
                i_args += 1
            else:
                self.unit = ''
              
            self.n_wl = len(self.wavelength)
            
            # initializing
            super(spc, self).__init__(spcData, 
                 index = np.arange(self.n_spc),
                 columns = self.__set_column_names('spc', self.n_wl)
                 )
            
            # adding of labels
            for key in kwargs:
                self.add_label(key, kwargs[key])
                
            i_label = 0
            while i_args+i_label < len(args):
                key = ''.join(['unnamed_', self.__get_unnamed_nr()])
                self.add_label(key, args[i_args+i_label])
                i_label += 1
            """
    def __get_unnamed_nr(self):
        stop = False
        i_unnamed = 1
        while stop == False:
            if i_unnamed in self._reserved_attr['columns']['unnamed']:
                i_unnamed += 1
            else:
                self._reserved_attr['columns']['unnamed'] = np.append( self._reserved_attr['columns']['unnamed'], i_unnamed)
                stop = True
        return str(i_unnamed)
            
    def __setattr__(self, name, value):
        if name.startswith('_'):
            pd.DataFrame.__setattr__(self, name, value)
        elif name in self._reserved_attr:
            self._reserved_attr[name] = value
        elif name in self._reserved_attr['columns']:
            self.__setitem__(name, value)
        elif not name.startswith('_'):
            self._custom_attr[name] = value
            pd.DataFrame.__setattr__(self, name, value)
        else:
            pd.DataFrame.__setattr__(self, name, value)
            
    
    def __getattr__(self, name):
        print(name)
        return 0
        if name.startswith('_'):
            return pd.DataFrame.__getattr__(self, name)
        elif name in self._metadata:
            return pd.DataFrame.__getattr__(self, name)
        elif name in self._reserved_attr:
            return self._reserved_attr[name]
        elif name in self._custom_attr:
            return self._custom_attr[name]
        elif name in self._reserved_attr['columns']:
            key_list = self._reserved_attr['columns'][name]
            val = pd.DataFrame.__getitem__(self, key = key_list).values
            if np.array(self._reserved_attr['columns'][name]).size == 1:
                return np.ravel(val)
            else:
                return val
        else:
            return pd.DataFrame.__getattr__(self, name)
    
    def __hasattr__(self, name):
        if name in self._metadata:
            return pd.DataFrame.__hasattr__(self, name)
        elif name in self._reserved_attr:
            return True
        elif name in self._custom_attr:
            return True
    
    def __delattr__(self, name):
        if name in self._metadata:
            print('Attribute can´t be removed!')
            return None
        elif name in self._reserved_attr:
            print('Attribute can´t be removed!')
            return None
        elif name in self._custom_attr:
            return self._custom_attr.pop(name)
        else:
            print('No such attribute!')
            return None  
    
    def __getitem__(self, key):
        if isinstance(key, tuple):
            wl = key[1]
            key = key[0]
            if isinstance(wl, slice):
                start = wl.start
                stop = wl.stop
                flag = np.arange(self.n_wl)
                flag[self.wavelength<start] = -1
                flag[self.wavelength>stop] = -1
                flag = flag[flag != -1]
                
                start = min(flag)
                stop = max(flag)+1
                
                new_wl = self.wavelength[start:stop]
                new_spc = self.spc[:,start:stop]
                                
                obj = copy.deepcopy(self)
                obj._custom_attr = copy.deepcopy(self._custom_attr)
                obj._reserved_attr = copy.deepcopy(self._reserved_attr)
                
                obj.spc = new_spc
                obj.wavelength = new_wl
                obj.n_wl = len(new_wl)
                return obj
        if not isinstance(key, tuple) and key in self._reserved_attr['columns']:
            key_list = self._reserved_attr['columns'][key]
            return pd.DataFrame.__getitem__(self, key = key_list)
    
    def __delitem__(self, key):
        if not isinstance(key, tuple) and key in self._reserved_attr['columns']:
            key_list = self._reserved_attr['columns'][key]
            del self._reserved_attr['columns'][key]
            for key in key_list:
                pd.DataFrame.__delitem__(self, key)
    
    def __setitem__(self, key, value):
        if not isinstance(key, tuple) and key in self._reserved_attr['columns']:
            key_list = self._reserved_attr['columns'][key]
            if np.array(value).shape == (self.n_spc, len(key_list)):
                pd.DataFrame.__setitem__(self, key_list, value)
            else:
                self.del_label(key)
                self.add_label(key, value)

    def __set_column_names(self, name, length):
        if length == 1:
            my_column = [name]
        else:
            my_column = [''.join([name, '_', str(col_nr)]) for col_nr in range(length)]
        
        self._reserved_attr['columns'][name] = my_column
        return my_column
    
    def __show_label(self, label):
        my_len = label.shape[0]
        if my_len > 10:
            my_label = list(map(str, label))
            my_label = np.append(my_label[:3], my_label[-3:])
            my_label = [(lambda i,k : k +', ' if i < 5 else k)(i,k) for i,k in enumerate(my_label)]
            my_label.insert(3,'..., ')
            my_label =''.join(my_label)
        else:
            my_label = list(map(str, label))
            my_label = [(lambda i,k : k +', ' if i < my_len-1 else k)(i,k) for i,k in enumerate(my_label)]
            my_label =''.join(my_label)
        return my_label
    
    def __return_unit(self):
        if self.unit == '':
            return ''
        else:
            return ' ['+self.unit+']'
    
    def __show_attr(self):
        
        if len(self._custom_attr)>0:           
            return ''.join([''.join(['  ',key,': ', str(self._custom_attr[key]), '\n']) for key in self._custom_attr])
        else:
            return '  none\n'
    
    def __list_labels(self):
        my_names = self._reserved_attr['columns'].copy()
        del my_names['unnamed']
        del my_names['spc']
        
        
        if len(my_names) != 0:
            return ''.join([''.join(['  ',name, ': ', self.__show_label(self.__getattr__(name)), '\n']) for name in my_names])
        else:
            return '  none'
        
    
    def __repr__(self):
        
        return  ''.join([
                'SpectralAnalysisPackage spc object'
                '\nnumber of spectra: %s',
                '\nspectral channels: %s',
                '\nwavelength', self.__return_unit(), ': ', self.__show_label(self.wavelength),
                '\nattributes:\n',
                self.__show_attr(),
                'labels:',
                '\n',self. __list_labels()                
                ]) % (self.n_spc, self.n_wl) 
    
    def add_label(self, key, val, p = False):
        val = np.array(val)
        if val.shape == ():
            val = np.full(self.n_spc, val)
        elif val.size == 1:
            val = np.full(self.n_spc, np.ravel(val)[0])
        elif val.shape[0] != self.n_spc and val.ndim == 1:
            val = np.tile(val, (self.n_spc,1))
        
        if p == True:
            print(self.__repr__)
        #else val.shape[0] == self.n_spc and val.ndim == 1:
            
        #else:
        #    raise ValueError('Length of label values is not equal to number of spectra')
                     
        if val.ndim == 1:
            self.__set_column_names(key, 1)
            frame = pd.DataFrame(data = val, columns = [key], index = self.index.values)
            super(spc, self).__init__(pd.concat([self, frame], sort = False, axis=1))
        else:
            key_list = self.__set_column_names(key, val.shape[1])
            frame = pd.DataFrame(data = val, columns = key_list, index = self.index.values)
            super(spc, self).__init__(pd.concat([self, frame], sort = False, axis=1))
    
    def del_label(self, key, p = False):
        self.__delitem__(key)
        if p == True:
            print(self.__repr__)
        

        
        
            
            
    
    

"""
"""

import numpy as np
import pandas as pd
import copy    

class spc(pd.DataFrame):
    
    _custom_attr = dict()
    _reserved_attr = dict()

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return spc(*args, **kwargs).__finalize__(self)
        return _c
    
    def __init__(self, *args, **kwargs):
        if isinstance(args[0], (pd.DataFrame, pd.core.internals.BlockManager)):
            #print(1)
            super(pd.DataFrame, self).__init__(args[0])
        else:
            
            # dict for reserved attributes
            
            self._reserved_attr = {
                    'spc': None,
                    'wavelength': None,
                    'unit': None,
                    'n_spc': None,
                    'shape_spc': None,
                    'n_wl': None
                    }
            
            i_args = 0
            
            # setting the spc attribute
            if 'spc' in kwargs:
                self.spc = np.array(kwargs.pop('spc', None))
            elif len(args)-i_args > 0:
                self.spc = np.array(args[i_args])
                i_args += 1
            else:
                raise ValueError('spc must be given!')
             
            # setting the wavelength attribute
            if 'wavelength' in kwargs:
                self.wavelength = np.array(kwargs.pop('wavelength', None))
            elif len(args)-i_args > 0:
                self.wavelength = np.array(args[i_args])
                i_args += 1
            else:
                self.wavelength = np.arange(self.spc.shape[-1])
            if self.spc.shape[-1] != self.wavelength.shape[0]:
                raise ValueError('Wavelength dimension did not fit spectral dimension.\n Expected %s channels, received %s.' % (self.spc.shape[-1],self.wavelength.shape[0]))
            
            # setting the unit attribute
            if 'unit' in kwargs:
                self.unit = str(kwargs.pop('unit', None))
            elif len(args)-i_args > 0:
                self.unit = str(args[i_args])
                i_args += 1
            else:
                self.unit = ''
              
            self.shape_spc = np.array(self.spc.shape[:-1])
            self.n_spc = np.prod(self.spc.shape[:-1])
            self.n_wl = len(self.wavelength)
            
            # dict for custom attributes
            
            # initializing
            super(spc, self).__init__(index = np.arange(self.n_spc))
            
            # adding of labels
            for key in kwargs:
                self.add_label(key, kwargs[key])
                
            i_label = 0
            while i_args+i_label < len(args):
                key = ''.join(['unnamed #', str(i_label+1)])
                self.add_label(key, args[i_args+i_label])
                i_label += 1
            
    def __setattr__(self, name, value):
        if name in self._reserved_attr:
            self._reserved_attr[name] = value
        elif name in self._custom_attr:
            self._custom_attr[name] = value
        else:
            pd.DataFrame.__setattr__(self, name, value)
    
    def __getattr__(self, name):
        if name in self._metadata:
            return pd.DataFrame.__getattr__(self, name)
        elif name in self._reserved_attr:
            return self._reserved_attr[name]
        elif name in self._custom_attr:
            return self._custom_attr[name]
    
    def __hasattr__(self, name):
        if name in self._metadata:
            return pd.DataFrame.__hasattr__(self, name)
        elif name in self._reserved_attr:
            return True
        elif name in self._custom_attr:
            return True
    
    def __delattr__(self, name):
        if name in self._metadata:
            print('Attribute can´t be removed!')
            return None
        elif name in self._reserved_attr:
            print('Attribute can´t be removed!')
            return None
        elif name in self._custom_attr:
            return self._custom_attr.pop(name)
        else:
            print('No such attribute!')
            return None
    
    def __return_unit(self):
        if self.unit == '':
            return ''
        else:
            return ' ['+self.unit+']'
    
    def __show_label(self, label):
        my_len = label.shape[0]
        if my_len > 10:
            my_label = np.append(label[:3], label[-3:])
            my_label = list(map(str, my_label))
            my_label = [(lambda i,k : k +', ' if i < 5 else k)(i,k) for i,k in enumerate(my_label)]
            my_label.insert(3,'..., ')
            my_label =''.join(my_label)
        else:
            my_label = list(map(str, label))
            my_label = [(lambda i,k : k +', ' if i < my_len-1 else k)(i,k) for i,k in enumerate(my_label)]
            my_label =''.join(my_label)
        return my_label
    
    def __show_attr(self):
        
        if len(self._custom_attr)>0:           
            return ''.join([''.join(['  ',key,': ', str(self._custom_attr[key]), '\n']) for key in self._custom_attr])
        else:
            return '  none\n'
    
    def __list_labels(self):
        my_names = list(self.columns.values)
        if len(my_names) != 0:
            return ''.join([''.join(['  ',name, ': ', self.__show_label(self[name]), '\n']) for name in my_names])
        else:
            return '  none'
    
    def __repr__(self):
        
        return  ''.join([
                'SpectralAnalysisPackage spc object'
                '\nnumber of spectra: %s',
                '\nspectral channels: %s',
                '\nwavelength', self.__return_unit(), ': ', self.__show_label(self.wavelength),
                '\nattributes:\n',
                self.__show_attr(),
                'labels:',
                '\n',self. __list_labels()                
                ]) % (self.n_spc, self.n_wl) 
    
    def __getitem__(self,key):
        if isinstance(key, tuple):
            wl = key[1]
            key = key[0]
            if isinstance(wl, slice):
                start = wl.start
                stop = wl.stop
                flag = np.arange(self.n_wl)
                flag[self.wavelength<start] = -1
                flag[self.wavelength>stop] = -1
                flag = flag[flag != -1]
                
                start = min(flag)
                stop = max(flag)+1
                
                new_wl = self.wavelength[start:stop]
                new_spc = self.spc[:,start:stop]
                                
                obj = copy.deepcopy(self)
                obj._custom_attr = copy.deepcopy(self._custom_attr)
                obj._reserved_attr = copy.deepcopy(self._reserved_attr)
                
                obj.spc = new_spc.__copy__()
                obj.wavelength = new_wl.__copy__()
                return obj
                
        else:
            obj = copy.deepcopy(self)
            obj._custom_attr = copy.deepcopy(self._custom_attr)
            obj._reserved_attr = copy.deepcopy(self._reserved_attr)
            
        if isinstance(key, int):
            new_spc = obj.spc[key,:]
            
            obj = pd.DataFrame.__getitem__(obj,slice(key,key+1))
            obj._custom_attr = copy.deepcopy(self._custom_attr)
            obj._reserved_attr = copy.deepcopy(self._reserved_attr)
            obj.spc = new_spc
            obj.n_spc = 1
            obj.shape_spc = np.array([1])
            
            return obj
        else:
            return pd.DataFrame.__getitem__(self,key)
    
    #def __setitem__(self,key, val):
    #    pd.DataFrame.__setitem__(self, key, val)
    
   
        
    def add_label(self, key, val):
        
        if np.array(val).shape == ():
            val = np.full(self.shape[0], val)
        elif np.array(val).size == 1:
            val = np.full(self.shape[0], np.ravel(val)[0])
        elif val.size != self.shape[0]:
            raise ValueError('Length of label values is not equal to number of spectra')
            
        kwargs = {key : pd.Series(val)}
        super(spc, self).__init__(self.assign(**kwargs))
 
        
   
class spc(pd.DataFrame):
    
    _metadata = ['custom_attr', 'reserved_attr']
    _myattr = 'attr'

    @property
    def _constructor(self):
    
        def _c(*args, **kwargs):
            return spc(*args, **kwargs).__finalize__(self)
        return _c
    def __init__(self, *args, **kwargs):
        if isinstance(args[0], (pd.DataFrame, pd.core.internals.BlockManager)):
            super(pd.DataFrame, self).__init__(args[0])
        else:
            
            # dict for reserved attributes
            self._myattr = 5   
            self.reserved_attr = {
                    'spc': None,
                    'wavelength': None,
                    'unit': None,
                    'n_spc': None,
                    'shape_spc': None,
                    'n_wl': None
                    }
            
            i_args = 0
            
            # setting the spc attribute
            if 'spc' in kwargs:
                self.spc = np.array(kwargs.pop('spc', None))
            elif len(args)-i_args > 0:
                self.spc = np.array(args[i_args])
                i_args += 1
            else:
                raise ValueError('spc must be given!')
             
            # setting the wavelength attribute
            if 'wavelength' in kwargs:
                self.wavelength = np.array(kwargs.pop('wavelength', None))
            elif len(args)-i_args > 0:
                self.wavelength = np.array(args[i_args])
                i_args += 1
            else:
                self.wavelength = np.arange(self.spc.shape[-1])
            if self.spc.shape[-1] != self.wavelength.shape[0]:
                raise ValueError('Wavelength dimension did not fit spectral dimension.\n Expected %s channels, received %s.' % (self.spc.shape[-1],self.wavelength.shape[0]))
            
            # setting the unit attribute
            if 'unit' in kwargs:
                self.unit = str(kwargs.pop('unit', None))
            elif len(args)-i_args > 0:
                self.unit = str(args[i_args])
                i_args += 1
            else:
                self.unit = ''
              
            self.shape_spc = np.array(self.spc.shape[:-1])
            self.n_spc = np.prod(self.spc.shape[:-1])
            self.n_wl = len(self.wavelength)
            
            # dict for custom attributes
                           
            self.custom_attr = dict()
            
            # initializing
            super(spc, self).__init__(index = np.arange(self.n_spc))
            
            # adding of labels
            for key in kwargs:
                self.add_label(key, kwargs[key])
                
            i_label = 0
            while i_args+i_label < len(args):
                key = ''.join(['unnamed #', str(i_label+1)])
                self.add_label(key, args[i_args+i_label])
                i_label += 1
            
    def __setattr__(self, name, value):
        if not name in self._metadata and not name.startswith('_'):
            if name == '_myattr':
                self._myattr = value
            elif not name in self.reserved_attr:
                self.custom_attr[name] = value
            else:
                self.reserved_attr[name] = value
        else:
            pd.DataFrame.__setattr__(self, name, value)
    
    def __getattr__(self, name):
        if name == '_myattr':
            return self._myattr
        elif name in self.reserved_attr:
            return self.reserved_attr[name]
        elif name in self.custom_attr:
            return self.custom_attr[name]
        else:
            return pd.DataFrame.__getattr__(self, name)
    
    def __hasattr__(self, name):
        if name in self.reserved_attr:
            return True
        elif name in self.custom_attr:
            return True
        else:
            return pd.DataFrame.__hasattr__(self, name)
    
    def __delattr__(self, name):
        if name in self._metadata:
            print('Attribute can´t be removed!')
            return None
        elif name in self.reserved_attr:
            print('Attribute can´t be removed!')
            return None
        elif name in self.custom_attr:
            return self.custom_attr.pop(name)
        else:
            print('No such attribute!')
            return None
    
    def __return_unit(self):
        if self.unit == '':
            return ''
        else:
            return ' ['+self.unit+']'
    
    def __show_label(self, label):
        my_len = label.shape[0]
        if my_len > 10:
            my_label = np.append(label[:3], label[-3:])
            my_label = list(map(str, my_label))
            my_label = [(lambda i,k : k +', ' if i < 5 else k)(i,k) for i,k in enumerate(my_label)]
            my_label.insert(3,'..., ')
            my_label =''.join(my_label)
        else:
            my_label = list(map(str, label))
            my_label = [(lambda i,k : k +', ' if i < my_len-1 else k)(i,k) for i,k in enumerate(my_label)]
            my_label =''.join(my_label)
        return my_label
    
    def __show_attr(self):
        
        if len(self.custom_attr)>0:           
            return ''.join([''.join(['  ',key,': ', str(self.custom_attr[key]), '\n']) for key in self.custom_attr])
        else:
            return '  none\n'
    
    def __list_labels(self):
        my_names = list(self.columns.values)
        if len(my_names) != 0:
            return ''.join([''.join(['  ',name, ': ', self.__show_label(self[name]), '\n']) for name in my_names])
        else:
            return '  none'
    
    def __repr__(self):
        return  ''.join([
                'SpectralAnalysisPackage spc object'
                '\nnumber of spectra: %s',
                '\nspectral channels: %s',
                '\nwavelength', self.__return_unit(), ': ', self.__show_label(self.wavelength),
                '\nattributes:\n',
                self.__show_attr(),
                'labels:',
                '\n',self. __list_labels()                
                ]) % (self.n_spc, self.n_wl) 
    
    def __getitem__(self,key):
        if isinstance(key, tuple):
            wl = key[1]
            key = key[0]
            if isinstance(wl, slice):
                start = wl.start
                stop = wl.stop
                flag = np.arange(self.n_wl)
                flag[self.wavelength<start] = -1
                flag[self.wavelength>stop] = -1
                flag = flag[flag != -1]
                
                start = min(flag)
                stop = max(flag)+1
                
                new_wl = self.wavelength[start:stop]
                new_spc = self.spc[:,start:stop]
                
                print(id(new_spc), ' - ', id(self.spc))
                
                #obj = self.__deepcopy__()
                obj = copy.deepcopy(self)
                obj.spc = new_spc.__copy__()
                obj.wavelength = new_wl.__copy__()
                return obj
                
        else:
            obj = self.__copy__()
            
        if isinstance(key, int):
            
            #new_spc = obj.spc[key,:]
            
            obj = pd.DataFrame.__getitem__(obj,slice(key,key+1))
            
            #obj.spc = new_spc
            #obj.n_spc = 1
            #obj.shape_spc = np.array([1])
            
            return obj
        else:
            return pd.DataFrame.__getitem__(self,key)
    
    #def __setitem__(self,key, val):
    #    pd.DataFrame.__setitem__(self, key, val)
    
    todo for array-entrys:
    def __setitem__(self,key, val):
    def __getitem__(self,key):
    
        
    def add_label(self, key, val):
        
        if np.array(val).shape == ():
            val = np.full(self.shape[0], val)
        elif np.array(val).size == 1:
            val = np.full(self.shape[0], np.ravel(val)[0])
        elif val.size != self.shape[0]:
            raise ValueError('Length of label values is not equal to number of spectra')
            
        kwargs = {key : pd.Series(val)}
        super(spc, self).__init__(self.assign(**kwargs))
"""
"""
class spc(pd.DataFrame):
    

            
            
                
        
    
        if isinstance(key, tuple):
            wl = key[1]
            key = key[0]
            if isinstance(wl, slice):
                start = wl.start
                stop = wl.stop
                flag = np.arange(len(self.wavelength))
                flag[self.wavelength<start] = -1
                flag[self.wavelength>stop] = -1
                flag = flag[flag != -1]
                
                start = min(flag)
                stop = max(flag)+1
                
                new_wl = self.wavelength[start:stop]
                new_spc = self.__getitem__('spc')[:,start:stop]
                
                obj = self.__copy__()
                obj['spc'] = new_spc
                obj.wavelength = new_wl
        else:
            obj = self
        
        if isinstance(key, int):
            return pd.DataFrame.__getitem__(obj,slice(key,key+1))
        elif isinstance(key, str) and key == 'spc':
            spc_old = np.array(pd.DataFrame.__getitem__(obj, key))
            spc_new = np.zeros((spc_old.size, obj.wavelength.size))
            
            for i, my_spc in zip(np.arange(spc_old.size), spc_old):  # where must be a better way!
                spc_new[i,:] = my_spc
            return spc_new
        else:
            return pd.DataFrame.__getitem__(obj,key)
    
    
            

        
    
"""
        
   

"""
class spc(pd.DataFrame):
    
    _metadata = ['wavelength', 'unit', 'custom_attr']

    
    @property
    def _constructor(self):
        
        def _c(spcData, *args, **kwargs):
            return spc(spcData, *args, **kwargs).__finalize__(self)
        return _c
    
    def __init__(self, spcData, *args, **kwargs):
        if isinstance(spcData, (pd.DataFrame, pd.core.internals.BlockManager)):
            super(spc, self).__init__(spcData)
        else:
            # formatting of spcData
            n_spc = np.prod(spcData.shape[0:-1])
            enum = np.arange(n_spc)
            data = pd.concat([
                    pd.DataFrame([[spc]], columns = {'spc'}, index = {ind})
                    for spc, ind in zip(spcData, enum)])
            
            i_args = 0
            
            # setting the wavelength attribute
            if 'wavelength' in kwargs:
                self.wavelength = np.array(kwargs.pop('wavelength', None))
            elif len(args)-i_args > 0:
                self.wavelength = np.array(args[i_args])
                i_args += 1
            else:
                self.wavelength = np.arange(spcData.shape[-1])
            if spcData.shape[-1] != self.wavelength.shape[0]:
                raise ValueError('Wavelength dimension did not fit spectral dimension.\n Expected %s channels, received %s.' % (spcData.shape[-1],self.wavelength.shape[0]))
            
            # setting the unit attribute
            if 'unit' in kwargs:
                self.unit = str(kwargs.pop('unit', None))
            elif len(args)-i_args > 0:
                self.unit = str(args[i_args])
                i_args += 1
            else:
                self.unit = ''
            
            # dict for custom attributes
            
            self.custom_attr = dict()
            
            # initializing
            super(spc, self).__init__(data)
            
            # adding of labels
            for key in kwargs:
                self.add_label(key, kwargs[key])
                
            i_label = 0
            while i_args+i_label < len(args):
                key = ''.join(['unnamed #', str(i_label+1)])
                self.add_label(key, args[i_args+i_label])
                i_label += 1
            
            
                
        
    def __getitem__(self,key):
        if isinstance(key, tuple):
            wl = key[1]
            key = key[0]
            if isinstance(wl, slice):
                start = wl.start
                stop = wl.stop
                flag = np.arange(len(self.wavelength))
                flag[self.wavelength<start] = -1
                flag[self.wavelength>stop] = -1
                flag = flag[flag != -1]
                
                start = min(flag)
                stop = max(flag)+1
                
                new_wl = self.wavelength[start:stop]
                new_spc = self.__getitem__('spc')[:,start:stop]
                
                obj = self.__copy__()
                obj['spc'] = new_spc
                obj.wavelength = new_wl
        else:
            obj = self
        
        if isinstance(key, int):
            return pd.DataFrame.__getitem__(obj,slice(key,key+1))
        elif isinstance(key, str) and key == 'spc':
            spc_old = np.array(pd.DataFrame.__getitem__(obj, key))
            spc_new = np.zeros((spc_old.size, obj.wavelength.size))
            
            for i, my_spc in zip(np.arange(spc_old.size), spc_old):  # where must be a better way!
                spc_new[i,:] = my_spc
            return spc_new
        else:
            return pd.DataFrame.__getitem__(obj,key)
    
    def __setitem__(self,key, val):
        if key == 'spc':
            n_spc = np.prod(val.shape[0:-1])
            enum = np.arange(n_spc)
            data = pd.concat([
                        pd.DataFrame([[spc]], columns = {'spc'}, index = {ind})
                        for spc, ind in zip(val, enum)])
    
            pd.DataFrame.__setitem__(self, 'spc', data)
        else:
            pd.DataFrame.__setitem__(self, key, val)
            
    def __setattr__(self, name, value):
        if not name in self._metadata and not name.startswith('_'):
            self.custom_attr[name] = value
        else:
            pd.DataFrame.__setattr__(self, name, value)
    
    def __delattr__(self, name):
        if name in self._metadata:
            print('Attribute can´t be removed!')
            return None
        elif name in self.custom_attr:
            return self.custom_attr.pop(name)
        else:
            print('No such attribute!')
            return None
        
    def __getattr__(self, name):
        if name in self.custom_attr:
            return self.custom_attr[name]
        else:
            return pd.DataFrame.__getattr__(self, name)
    
    def __hasattr__(self, name):
        if name in self.custom_attr:
            return True
        else:
            return pd.DataFrame.__hasattr__(self, name)
        
    def __return_unit(self):
        if self.unit == '':
            return ''
        else:
            return ' ['+self.unit+']'
    
    def __show_label(self, label):
        my_len = label.shape[0]
        if my_len > 10:
            my_label = np.append(label[:3], label[-3:])
            my_label = list(map(str, my_label))
            my_label = [(lambda i,k : k +', ' if i < 5 else k)(i,k) for i,k in enumerate(my_label)]
            my_label.insert(3,'..., ')
            my_label =''.join(my_label)
        else:
            my_label = list(map(str, label))
            my_label = [(lambda i,k : k +', ' if i < my_len-1 else k)(i,k) for i,k in enumerate(my_label)]
            my_label =''.join(my_label)
        return my_label
    
    def __show_attr(self):
        
        if len(self.custom_attr)>0:           
            return ''.join([''.join(['  ',key,': ', str(self.custom_attr[key]), '\n']) for key in self.custom_attr])
        else:
            return '  none\n'
        
    
    def __list_labels(self):
        my_names = list(self.columns.values)[1:]
        if len(my_names) != 0:
            return ''.join([''.join(['  ',name, ': ', self.__show_label(self[name]), '\n']) for name in my_names])
        else:
            return '  none'
        
    
    def __repr__(self):
        return  ''.join([
                'SpectralAnalysisPackage spc object'
                '\nnumber of spectra: %s',
                '\nspectral channels: %s',
                '\nwavelength', self.__return_unit(), ': ', self.__show_label(self.wavelength),
                '\nattributes:\n',
                self.__show_attr(),
                'labels:',
                '\n',self. __list_labels()                
                ]) % (self.shape[0], self.wavelength.size)    

    def add_label(self, key, val):
        
        if np.array(val).shape == ():
            val = np.full(self.shape[0], val)
        elif np.array(val).size == 1:
            val = np.full(self.shape[0], np.ravel(val)[0])
        elif val.size != self.shape[0]:
            raise ValueError('Length of label values is not equal to number of spectra')
            
        kwargs = {key : pd.Series(val)}
        super(spc, self).__init__(self.assign(**kwargs))
    
"""
        