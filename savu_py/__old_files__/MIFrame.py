import numpy as np
import pandas as pd
import copy    
import warnings

from operator import mul
from functools import reduce

from .__get_attribute__ import __get_attribute__
from .MISlice import MISlice

class MIFrame(pd.DataFrame):
    OPTIONS = {
        'show_attributes'       :  True,
        'show_labels'           :  True,
        'show_label_examples'   :  True,
        'n_row_example_max'     :  10,
        'n_row_example_pm'      :  3,
        'n_col_example_max'     :  10,
        'n_col_example_pm'     :  3
        }
    
    _metadata = ['_custom_attr','_reserved_attr']

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return MIFrame(*args, **kwargs).__finalize__(self)
        return _c
    
    def __init__(self, *args, **kwargs):
        try:
            is_frame = isinstance(args[0], (pd.DataFrame, pd.core.internals.BlockManager))
        except:
            is_frame = False
        if is_frame:
            super(MIFrame, self).__init__(args[0])
        else:
                        
            # dict for reserved attributes
            self._reserved_attr = {
                    'ga': __get_attribute__(parent = self),
                    'data_shape': None,
                    'unnamed_columns': np.array([]),
                    'n_dim': None,
                    'columns':None
                    }
            
            
            # dict for custom attributes
            self._custom_attr = dict()
                        
            # unpacking of args and kwargs:
            i_args = 0
            n_rows = 0
            
            # check, if n_rows/n_spc or data_shape in kwargs
            # if not, first of args or kwargs will define both!
            if 'data_shape' in kwargs:
                if 'n_rows' in kwargs:
                    kwargs.pop('n_rows')
                if 'n_spc' in kwargs:
                    kwargs.pop('n_spc')
                data_shape = kwargs.pop('data_shape')
                if isinstance(data_shape, int):
                    data_shape = [data_shape]
                if all(isinstance(x, int) for x in data_shape):
                    self.ga.data_shape = tuple(data_shape)
                    n_rows = reduce(mul, data_shape, 1)
                    self.ga.n_dim = len(data_shape)
            elif 'n_rows' in kwargs:
                if 'n_spc' in kwargs:
                    kwargs.pop('n_spc')
                n_rows = kwargs.pop('n_rows')
                if not isinstance(n_rows, int):
                    n_rows = 0
            elif 'n_spc' in kwargs:
                n_rows = kwargs.pop('n_spc')
                if not isinstance(n_rows, int):
                    n_rows = 0
            
            key = None
            # check, if data_shape is known, if not: load first kwargs or args
            if self.ga.n_dim == None or n_rows == 0:  # n_rows == 0 means one empty dimension --> this can't be
                if len(kwargs) != 0:
                    key = list(kwargs.keys())[0]
                    val = kwargs.pop(key)
                elif len(args) != 0:
                    key = 1 ############################################################################# do more here!
                    val = args[0]
                else:
                    raise ValueError('Not enough arguments to unpack!')
                
                shape = np.array(val).shape
                if n_rows == None or n_rows == 0:
                    if len(shape) > 1:
                        shape = shape[:-1]
                    self.ga.data_shape = tuple(shape)
                    n_rows = reduce(mul, shape, 1)
                    self.ga.n_dim = len(shape)
                else:
                    enum = np.array(list(enumerate(shape)))
                    prod = enum[0,1]
                    for i, v in enum[1:]:
                        prod *= v
                        enum[i,1] = prod
                    pos = enum[:,0][enum[:,1] == n_rows]
                    if len(pos) >= 1:           
                        pos = pos[-1]
                        shape = shape[:pos+1]
                    else:                           # if n_rows don't fit!
                        shape = shape[:-1]
                        warnings.warn('n_rows doesn`t fit to shape of data, calculating new n_rows')
                    self.ga.data_shape = tuple(shape)
                    n_rows = reduce(mul, shape, 1)
                    self.ga.n_dim = len(shape)
                        
            
            
            # creating the multidimensional array
            
            """
            prod = reduce(mul, shape, 1)
            index = np.reshape([np.reshape(ind,prod) for ind in np.indices(shape)], (len(shape),prod))
                this is a nice way to make an array where a[x,y,z] = [x,y,z]
            index = list(zip(*index))
            index = pd.MultiIndex.from_tuples(index)
            """
            
            if self.ga.n_dim > 1:
                index = [range(s) for s in self.ga.data_shape]
                index = pd.MultiIndex.from_product(index)
            else:
                index = np.arange(n_rows)
            
            
            rm_key = 'remove'
            if key != None:
                if key == 'remove':
                    rm_key = 'rm'
                    
            # initialize an empty data frame
            super(MIFrame, self).__init__(index = index, columns = pd.MultiIndex.from_product([[rm_key],['later']]))
           
            # adding of labels
            
            i_label = 0
            
            # the old one
            if key != None:
                if key != 1:
                    self.add_label(key, val)
                else:
                    key = 'unnamed'
                    self.add_label(key, val)
                    i_label += 1
            
            # remove placeholder for multiindexing
            super(MIFrame, self).__init__(self.drop(columns=rm_key))
            
            # the new ones
            
            for key in kwargs:
                self.add_label(key, kwargs[key])
            
            
            while i_args+i_label < len(args):
                #key = ''.join(['unnamed_', str(i_label+1)])
                key = 'unnamed'
                self.add_label(key, args[i_args+i_label])
                i_label += 1
            
            self.__update__()
        
    def __update__(self): 
        
        self.ga.columns = self.columns
    def __ga__(self, name):
        try:
            if name.startswith('_'):
                return pd.DataFrame.__getattr__(self, name)
            elif name in self._metadata:
                return pd.DataFrame.__getattr__(self, name)
            elif type(self._reserved_attr) != type(None):
                if name in self._reserved_attr:
                    return self._reserved_attr[name]
                elif name in self._custom_attr:
                    return self._custom_attr[name]
                elif name == 'n_spc' or name == 'n_rows':
                    return self.shape[0]
                else:
                    return pd.DataFrame.__getattr__(self, name)
            else:
                return pd.DataFrame.__getattr__(self, name)
        except:
            return None
    
    def __sa__(self, name, value):
        if name.startswith('_'):
            pd.DataFrame.__setattr__(self, name, value)
        elif name in self._metadata:
            pd.DataFrame.__setattr__(self, name, value)
        elif type(self._reserved_attr) != type(None):
            if name in self._reserved_attr:
                self._reserved_attr[name] = value
            else:
                self._custom_attr[name] = value
        else:
            pd.DataFrame.__setattr__(self, name, value)
    
    def __da__(self, name):
        if name.startswith('_'):
            print('Attribute can´t be removed!')
            return None
        if name in self._metadata:
            print('Attribute can´t be removed!')
            return None
        elif type(self._reserved_attr) != type(None):
            if name in self._reserved_attr:
                print('Attribute can´t be removed!')
                return None
            elif name == 'n_spc' or name == 'n_rows':
                print('Attribute can´t be removed!')
                return None
            elif name in self._custom_attr:
                value = self._custom_attr[name]
                del self._custom_attr[name]
                return value
        else:
            print('No such attribute!')
            return None
    
    def __ha__(self, name):
        if name.startswith('_'):
            return pd.DataFrame.__hasattr__(self, name)
        elif name in self._metadata:
            return pd.DataFrame.__hasattr__(self, name)
        elif type(self._reserved_attr) != type(None):
            if name in self._reserved_attr:
                return True
            elif name in self._custom_attr:
                return True
            elif name == 'n_rows' or name == 'n_spc':
                return True
        else:
            return False
        
    def __getattr__(self, name):
        if name == 'ga':
            #return object.__getattribute__(self, '_reserved_attr')['ga']
            return self.__ga__('ga')
        elif name.startswith('_'):
            return pd.DataFrame.__getattr__(self, name)
        
        elif name in self.get_labels():
            mi = self.get_col_idx(name)
            obj =  pd.DataFrame.__getitem__(self,mi)
            obj.__update__()
            return obj

        #else:
        #    return pd.DataFrame.__getattr__(self, name)
    
    def __setattr__(self, name, value):
        if name == 'ga':
            self.__sa__(name, value)
        elif name.startswith('_'):
            pd.DataFrame.__setattr__(self, name, value)
        
        else:
            self.__setitem__(name, value)
    
    def __hasattr__(self, name):
        if name == 'ga':
            return pd.DataFrame.__hasattr__(self, name)
        elif name.startswith('_'):
            return pd.DataFrame.__hasattr__(self, name)
        elif name in self.get_labels():
            return True
        else:
            return pd.DataFrame.__hasitem__(self, name)
    
    def __delattr__(self, name):
        if name == 'ga':
            print('Attribute can´t be removed!')
            return None
        elif name.startswith('_'):
            print('Attribute can´t be removed!')
            return None
        elif name in self.get_labels():
            obj =  self.__delitem__(name)
            #obj.__update__()
            return obj
        else:
            print('No such attribute!')
            return None
            
    def add_label(self, key, val):
        val = self.__reshape_data(np.array(val))
        
        if val.ndim == 1:
            key, key_list = self.__get_column_names(key, 1)
        else:
            key, key_list = self.__get_column_names(key, val.shape[1])
        
        columns = pd.MultiIndex.from_product([[key],key_list])
        new_frame = pd.DataFrame(val, columns = columns, index = self.index)  
        
        super(MIFrame, self).__init__(pd.concat([self, new_frame], axis=1, sort=False))
        
    def __reshape_data(self, value):
        value = np.array(value)
        if value.ndim == 0:
            value = np.full(self.shape[0], value)
        elif value.size == 1:
            value = np.full(self.shape[0], np.ravel(value)[0])
        elif value.ndim == 2:
            if value.shape[0] == self.shape[0]:
                pass
        elif value.ndim == 1 and value.size == self.shape[0]:
            pass
        elif (value.ndim == 1 and value.size != self.shape[0]) or (value.ndim == 2 and value.shape[0] == 1):
            value = np.ravel(value)
            value = np.full((self.shape[0], value.size), value)
        elif value.ndim == self.ga.n_dim and value.shape == self.ga.data_shape:
            value = np.reshape(value, self.shape[0])
        elif value.ndim == self.ga.n_dim+1 and value.shape[:self.ga.n_dim] == self.ga.data_shape:
            value = np.reshape(value, (self.shape[0], value.shape[-1]))
        else:
            raise ValueError('Dimensionality of label too high!')
        return value
    
    def __get_column_names(self, name, length):
        if name == 'unnamed':
            n_unnamed = self._reserved_attr['unnamed_columns'].size
            if n_unnamed == 0:
                n_index = 1
            else:
                new_ind = np.array(range(n_unnamed + 1))+1
                old_ind = self._reserved_attr['unnamed_columns']
                n_index = new_ind[np.where(np.array([ind in old_ind for ind in new_ind]) == False)[0][0]]
            name = ''.join(['unnamed_',str(n_index)])
            self._reserved_attr['unnamed_columns'] = np.append(self._reserved_attr['unnamed_columns'], n_index).astype(int)
        
        my_column = range(length)
        """
        if length == 1:
            my_column = [name]
        else:
            my_column = [''.join([name, '_', str(col_nr+1)]) for col_nr in range(length)]
        self._reserved_attr['columns'][name] = my_column
        """

        return name, my_column
    
    def get_labels(self):
        self.__update__()
        return list(self.ga.columns.get_level_values(0).drop_duplicates())
    
    def __find_key(self, key):
        keys = self.get_labels()
        n_findings = 0
        head = ''
        for name in keys:
            if name in key:
                n_findings += 1
                head = name
        if n_findings == 1:
            return head
        elif n_findings > 1:
            warnings.warn('Cannot identify correct column: multiple column heads contain this name!')
        else:
            warnings.warn('Cannot identify correct column: no column head contains this name!')
        return 0
    
    
    def __getitem__(self,key):
        #if not isinstance(key, tuple):
        #    key = (key,)
        """
            order of getitem:
                - direct use: MISlice, pd.MultiIndex
                - check if key is tuple: 
                    - no: do single stuff as befor
                    - yes: first element is string --> columns
                                         not string -> rows
        
        """
        #obj = self
        #obj = copy.deepcopy(self)
        obj = self.__copy__()
        obj._reserved_attr = copy.deepcopy(self._reserved_attr)
        obj._custom_attr = copy.deepcopy(self._custom_attr)
        
        new_ga = __get_attribute__(obj)
        obj.ga = new_ga
        obj.__update__()
        
        if isinstance(key, MISlice):
            try:
                obj = obj.loc[key.slice_matrix]
            except:
                warnings.warn('Key not in rows')
        elif isinstance(key, pd.MultiIndex):
            try:
                obj =  pd.DataFrame.__getitem__(obj, key)
            except:
                warnings.warn('Key not in rows')
        
        elif isinstance(key, int):
            obj =  obj.iloc()[key:key+1,:]
        
        elif isinstance(key, slice) or isinstance(key, list):
            obj =  obj.iloc()[key,:]
        
        elif isinstance(key, str):
            keys = obj.get_labels()
            if key in keys:
                mi = obj.get_col_idx(key)
                
                obj =  pd.DataFrame.__getitem__(obj,mi)
                
            else:
                warnings.warn('Item not found!')
                
        elif isinstance(key, tuple):
            #obj = copy.deepcopy(self)
            #obj._reserved_attr = copy.deepcopy(self._reserved_attr)
            #obj._custom_attr = copy.deepcopy(self._custom_attr)
            #new_ga = __get_attribute__(obj)
            #obj.ga = new_ga
            
            if isinstance(key[0], str):
                if len(key) == 2:
                    try:
                        if isinstance(key[1], (int, np.integer)):
                            mi = obj.get_col_idx(key[0])[key[1]]                            
                            obj = obj.__getitem__(pd.MultiIndex.from_arrays([[mi[0]],[mi[1]]]))
                        else:
                            obj = obj.__getitem__(obj.get_col_idx(key[0])[key[1]])
                    except:
                        warnings.warn('key error')
                else:
                    warnings.warn('key error')
            else:
                for i_key in key:
                    obj = obj.__getitem__(i_key)
            #return obj    
            """    
            i_tuple = 0
            n_tuple = len(key)
            if i_tuple < n_tuple:
                
                if isinstance(key[i_tuple], slice) or isinstance(key[i_tuple], list):
                    obj = self.iloc()[key[i_tuple],:]
                else:
                    obj = MIFrame(pd.DataFrame(self.iloc()[key[i_tuple]]).transpose())
                    obj._reserved_attr = copy.deepcopy(self._reserved_attr)
                    obj._custom_attr = copy.deepcopy(self._custom_attr)
                new_ga = __get_attribute__(obj)
                obj.ga = new_ga
                return obj
            """
            
        else:
            print(type(key))
            obj = pd.DataFrame.__getitem__(obj,key)
        obj.__update__()
        self.__update__()
        return obj
        """
        if isinstance(key, int):
            return pd.DataFrame.__getitem__(self,slice(key,key+1))
        elif isinstance(key, list) or isinstance(key, tuple):
            if all(isinstance(k, int) for k in key):
                return self.iloc()[key,:]
            
        elif isinstance(key, str):
            keys = self.get_labels()
            if key in keys:
                mi = self.get_col_idx(key)
                return pd.DataFrame.__getitem__(self,mi)
                
            else:
                warnings.warn('Item not found!')
                
        else:
            return pd.DataFrame.__getitem__(self,key)
        """
        """ 
        elif not isinstance(key, slice):
            if not isinstance(key, pd.Index):
                print(type(key))
                if key in self.get_labels():
                    print()
                    #return self.loc[:, pd.IndexSlice[key, :]]
                    try:
                        idx = pd.IndexSlice
                        idx[:,  self.__find_key(key)]
                        return self.loc[idx,:]
                        #print( pd.DataFrame.__getitem__(self, )
                    except:
                        return '1'
            else:
                
                frames = [pd.DataFrame.__getitem__(self,pd.IndexSlice[k[0], k[1]]) for k in key]
                newFrame = pd.DataFrame(frames).T
                newFrame.columns = key
                
                newFrame = MIFrame(newFrame)
                newFrame._custom_attr = self._custom_attr
                newFrame._reserved_attr = self._reserved_attr
                
                return newFrame
            """
        #### include multiindex for rows!
        
    def __delitem__(self, key):
        self.__update__()
        
        if key in self.get_labels():
            mi = self.get_col_idx(key)
            super(MIFrame, self).__init__(self.drop(columns = mi))
        elif isinstance(key, slice) or isinstance(key, int):
            #key = list(range(key.stop)[key])
            key = self.index[key]
            super(MIFrame, self).__init__(pd.DataFrame.drop(self, key))
        else:
            super(MIFrame, self).__init__(pd.DataFrame.__delitem__(self, key))
    
    def __setitem__(self, key, value):
        value = self.__reshape_data(np.array(value))
        val_shape = np.array(value).shape
        
        if key in self.get_labels():
            key_list = self.get_col_idx(key)
            n_keys = len(key_list)
            
            if len(val_shape) == 2:
                if val_shape[1] == n_keys:
                    pd.DataFrame.__setitem__(self, key = key_list, value = value)
                else:
                    self.__delitem__(key)
                    self.add_label(key, value)
            else:
                if n_keys == 1:
                        pd.DataFrame.__setitem__(self, key = key_list, value = value)
                else:
                    self.__delitem__(key)
                    self.add_label(key, value)
         
        else:
            self.add_label(key, value)
        
    
    
    def __num_format(self, num, prec):
        if isinstance(num, (int, float)):
            s = "%.*E"%(prec, num)
            mantissa, exp = s.split('E')
            if int(exp) > 2 or int(exp) < -1:
                return s
            else:
                return "%.*f"%(prec, num)
        else:
            return str(num)
    
    def __list_labels(self):
        def _string_fun(number):
            return self.__num_format(number, 3)
        
        my_names = self.get_labels()
        rows = self.shape[0]
        
        if len(my_names) > 0:
            # first: look for the rows
            
            rowlim = self.OPTIONS['n_row_example_max']
            rowex = self.OPTIONS['n_row_example_pm']
            
            
            if 2*rowex > rowlim:
                rowex = np.amax([1, (np.ceil(rowlim/2)-1).astype(int)])
            if rowlim == 1:
                r_list = list([0,])    
            elif rows > rowlim:
                r_list = list(np.concatenate([
                              np.arange(rowex),
                              np.arange(rowex)-rowex
                           ], axis = 0))
            else:
                r_list = list(np.arange(rows))
            my_vals = self[r_list]
            
            
            my_str = np.array([])
            for name in my_names:
                key_list = self.get_col_idx(name)
                n_keys = len(key_list)
                if n_keys > 1 and self.OPTIONS['n_col_example_max'] != 1:
                    if rowlim > 0:
                        my_str = np.append(my_str, [''.join(['  ',name, ':\n'])])
                    else:
                        my_str = np.append(my_str, [''.join(['  ',name, '\n'])])
                    
                    collim = self.OPTIONS['n_col_example_max']
                    colex = self.OPTIONS['n_col_example_pm']
                    
                    if n_keys > collim:
                        if 2*colex > collim:
                            colex = (np.ceil(collim/2)-1).astype(int)
                        c_list = list(np.concatenate([
                              np.arange(colex),
                              np.arange(colex)-colex
                           ], axis = 0))
                        values = my_vals[(name,c_list)].values
                        idx_list = list(map(str,list(np.concatenate([
                              np.arange(colex),
                              np.arange(n_keys-colex, n_keys)
                           ], axis = 0))))
                    else:
                        values = my_vals[name].values
                        idx_list = list(map(str,list(range(n_keys))))
                else:
                    
                    if rowlim > 0:
                        my_str = np.append(my_str, [''.join(['  ',name, ': '])])
                    else:
                        my_str = np.append(my_str, [''.join(['  ',name])])
                    idx_list = [' ']
                    if self.OPTIONS['n_col_example_max'] == 1:
                        values = my_vals[name, list([0,])].values
                    else:
                        values = my_vals[name].values
                for i_col, index in enumerate(idx_list):
                    my_label = list(map(_string_fun,values[:,i_col]))
                    if rowlim == 0:
                        my_label = ''
                    elif rowlim == 1:
                        my_label = my_label[0]
                    elif rows > rowlim :
                        my_label = [(lambda i,k : k +', ' if i < 2*rowex else k)(i,k) for i,k in enumerate(my_label)]
                        my_label.insert(rowex,'..., ')
                        my_label = ''.join(my_label)[:-2]
                    else:
                        my_label = [(lambda i,k : k +', ' if i < rows-1 else k)(i,k) for i,k in enumerate(my_label)]
                        my_label = ''.join(my_label)[:-2]
                         
                    my_label =''.join(my_label)
                    if len(idx_list) == 1:
                        my_str = np.append(my_str, [''.join([my_label, '\n'])])
                    elif rowlim > 0:
                        my_str = np.append(my_str, [''.join(['   ',index, ': ', my_label, '\n'])])
                    
            my_str = ''.join(my_str)
        else:
            my_str = '  none\n'

        return my_str
    
    def __show_attr(self):
        
        if len(self._custom_attr)>0:
            return ''.join([''.join(['  ',key,': ', str(self._custom_attr[key]), '\n']) for key in self._custom_attr])
        else:
            return '  none\n'

    def __repr__(self):
        
        if self.OPTIONS['show_attributes'] == True:
            str_attr = ''.join([
                '\nattributes:\n',
                self.__show_attr()
                ])
        else:
            str_attr = ''
            
        if self.OPTIONS['show_labels'] == True:
            str_lab = ''.join([
                'labels:',
                '\n',self. __list_labels()     
                ])
        else:
            str_lab = ''
        
        return  ''.join([
                'SpectralAnalysisPack MIFrame object',
                '\nnumber of rows: %s',
                str_attr,
                str_lab,
                ]) % (self.shape[0]) 
        
    
    def get_col_idx(self, key):
        key_list = self.ga.columns.get_level_values
        key_list = key_list(1)[key_list(0) == key]
        key_list = pd.MultiIndex.from_product([[key], key_list])
        
        return key_list
    