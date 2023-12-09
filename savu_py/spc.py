import numpy as np
import pandas as pd
import copy    
import warnings

from .__get_attribute__ import __get_attribute__
from .MISlice import MISlice

class spc(pd.DataFrame):
    OPTIONS = {
        'show_attributes'       :  True,
        'show_labels'           :  True,
        'show_label_examples'   :  True,
        'n_row_example_max'     :  10,
        'n_row_example_pm'      :  3,
        'n_col_example_max'     :  10,
        'n_col_example_pm'      :  3
        }
    
    _metadata = ['_custom_attr','_reserved_attr','_pd_attr']
    
    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return spc(*args, **kwargs).__finalize__(self)
        return _c
    
    def __init__(self, *args, **kwargs):
        
        try:
            is_frame = isinstance(args[0], (pd.DataFrame, pd.core.internals.BlockManager))
        except:
            is_frame = False
        
        if is_frame:
            super(spc, self).__init__(args[0])
        else:
            ga = __get_attribute__(parent = self)
            #dict for reserved attributes            
                        
            self._reserved_attr = {
                    'wavelength': None,                     # wavelength axis of spc                      
                    'unit_wl': None,                        # wl-unit (e.g. eV, a.U.)
                    'unit_spc': None,                       # wl-unit (e.g. eV, a.U.)
                    'n_spc': None,                          # number of spc, equals length
                    'data_shape': None,                     # if possible, nD-shape of spc
                    'n_wl': None,                           # amount of channels
                    'ga': ga,                               # getattribute subclass
                    'unnamed_columns': np.array([]),        # dict to store unnamed columns
                    'n_dim': None,                          # n_dim of data shape, amount of multiindizes
                    'columns': None,                        # list of names of columns
                    'is_spc': True,                         # object has spc columns? if false --> object is label-like object
                    'is_empty': False,                      # applies only if initialized with no data
                    'pixsize': None                         # pixelsize
                    }
            # dict for custom attributes
            self._custom_attr = dict()
            
            # dict for pd attributes
            self._pd_attr = ['index']
            
            # unpacking of args and kwargs:
            i_args = 0
            n_rows = 0
            
            # setting the spc attribute
            if 'spc' in kwargs:
                self.ga.is_spc = True
                spcData = np.array(kwargs.pop('spc', None))
                kwargs.pop('is_spc', False)
                key_spcData = 'spc'
            else:
                is_spc = kwargs.pop('is_spc', True)
                if is_spc:
                    self.ga.is_spc = True
                    key_spcData = 'spc'
                else:
                    self.ga.is_spc = False
                    key_spcData = 'unnamed'
                    
                keys_reserved = list(self.ga._reserved_attr.keys())
                keys_kwargs = list(kwargs.keys())
                keys_unique = [x for x in keys_kwargs if not x in keys_reserved]
                if len(keys_unique)>0:
                    key_spcData = keys_unique[0]
                    spcData = np.array(kwargs.pop(key_spcData, None))
                elif (len(args)-i_args > 0):
                    spcData = np.array(args[i_args])
                    i_args += 1
                else:
                    spcData = np.array([])
                    self.ga.is_spc = False
                    self.ga.is_empty = True
                    #raise ValueError('data must be given!')
                
            ## what happens if spc_data is a 1D list? --> each element is new spc by default
            if spcData.ndim == 1:
                spcData = np.reshape(spcData, (spcData.shape[0],1))
            ## other way arround: spcData = [spcData]
            
            data_shape = np.array(spcData.shape)[:-1]
            n_rows = np.prod(data_shape)
            
            self.ga.n_dim = len(data_shape)
            self.ga.data_shape = tuple(data_shape)
            
            # setting the wavelength attribute
            if 'wavelength' in kwargs:
                wavelength = np.array(kwargs.pop('wavelength', None))
            elif len(args)-i_args > 0:
                wavelength = np.array(args[i_args])
                i_args += 1
            else:
                wavelength = np.arange(spcData.shape[-1])
            if spcData.shape[-1] != wavelength.size:
                raise ValueError('Wavelength dimension did not fit spectral dimension.\n Expected %s channels, received %s.' % (spcData.shape[-1], wavelength.size))
                  
            # setting the unit attribute
            if 'unit_wl' in kwargs:
                unit_wl = str(kwargs.pop('unit_wl', None))
            elif len(args)-i_args > 0:
                unit_wl = str(args[i_args])
                i_args += 1
            else:
                unit_wl = 'NA'
                
            # setting the unit attribute
            if 'unit_spc' in kwargs:
                unit_spc = str(kwargs.pop('unit_spc', None))
            elif len(args)-i_args > 0:
                unit_wl = str(args[i_args])
                i_args += 1
            else:
                unit_spc = 'a.U.'
                
            key = None
                        
            # creating the multidimensional array
            
            if self.ga.n_dim > 1:
                index = [range(s) for s in self.ga.data_shape]
                index = pd.MultiIndex.from_product(index)
            else:
                index = np.arange(n_rows)
            
            # initialize an empty data frame
        
            if spcData.ndim == 1:
                key, key_list = self.__get_column_names(key_spcData, 1)
            else:
                key, key_list = self.__get_column_names(key_spcData, spcData.shape[-1])
            
            columns = pd.MultiIndex.from_product([[key],key_list])
            
            ra = self._reserved_attr
            ca = self._custom_attr
            pa = self._pd_attr
            
            
            super(spc, self).__init__(np.reshape(spcData, (n_rows, spcData.shape[-1]) ), index = index, columns = columns)
            self.index.names = range(len(self.index.names))
            
            self._reserved_attr = ra
            self._custom_attr = ca
            self._pd_attr = pa
            
            self.ga._reserved_attr['wavelength'] = wavelength
            self.ga._reserved_attr['n_wl'] = wavelength.size   
            self.ga._reserved_attr['unit_wl'] = unit_wl
            self.ga._reserved_attr['unit_spc'] = unit_spc
            
            # adding of labels
            
            i_label = 0
            # the new ones
            
            for key in kwargs:
                self.add_label(key, kwargs[key])
            
            while i_args+i_label < len(args):
                #key = ''.join(['unnamed_', str(i_label+1)])
                key = 'unnamed'
                self.add_label(key, args[i_args+i_label])
                i_label += 1
            
            if self.ga.is_empty:
                for k in self.get_labels():
                    del self[k]
            
            self.__update__()
    
    def __copy__(self):
        
        self.__update__()
        
        ra = copy.deepcopy(self._reserved_attr)
        ca = copy.deepcopy(self._custom_attr)
        pa = copy.deepcopy(self._pd_attr)
        
        #obj = pd.DataFrame.copy(self, deep = True)
        obj = pd.DataFrame.copy(self)
       
        obj._reserved_attr = ra
        obj._custom_attr = ca
        obj._pd_attr = pa
        
        new_ga = __get_attribute__(obj)
        obj.ga = new_ga
        
        obj.__update__()
        
        return obj
    
    """ include deep = True!
    def copy():
        return self.__copy__()
    """
    
    def __update__(self): 
        
        self.ga.columns = self.columns
        keys = self.get_labels(is_update=True)
        if 'spc' in keys:
            self.ga.is_spc = True
        else:
            self.ga.is_spc = False

            
    def __ga__(self, name):
        try:
            if name.startswith('_'):
                return pd.DataFrame.__getattr__(self, name)
            elif name in self._metadata:
                return pd.DataFrame.__getattr__(self, name)
            elif name in self._pd_attr:
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
        elif name in self._pd_attr:
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
        elif name in self._pd_attr:
            pd.DataFrame.__delattr__(self, name)
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
        elif name in self._pd_attr:
            True
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
        elif name in self._pd_attr:
            return pd.DataFrame.__getattr__(self, name)
        elif name.startswith('_'):
            return pd.DataFrame.__getattr__(self, name)
        elif name in self.get_labels():
            return self.__getitem__(name)

    
    def __setattr__(self, name, value):
        if name == 'ga':
            self.__sa__(name, value)
        elif name.startswith('_'):
            pd.DataFrame.__setattr__(self, name, value)
        elif name in self._pd_attr:
            pd.DataFrame.__setattr__(self, name, value)
        else:
            self.__setitem__(name, value)
    
    def __hasattr__(self, name):
        if name == 'ga':
            return True
        elif name.startswith('_'):
            return pd.DataFrame.__hasattr__(self, name)
        elif name in self._pd_attr:
            True
        elif name in self.get_labels():
            return True
        else:
            return pd.DataFrame.__hasattr__(self, name)
    
    def __delattr__(self, name):
        if name == 'ga':
            print('Attribute can´t be removed!')
            return None
        elif name.startswith('_'):
            print('Attribute can´t be removed!')
            return None
        elif name in self._pd_attr:
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
        
        super(spc, self).__init__(pd.concat([self, new_frame], axis=1, sort=False))
        
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
    
    def get_labels(self, is_update=False):
        if is_update:
            return list(self.ga.columns.get_level_values(0).drop_duplicates())
        else:
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
    
    def __get_wl_idx__(self, key, wl):
        if isinstance(key, slice):
            key = np.array(list(range(key.stop)[key]))
            min_x = key[0]
            max_x = key[-1]
            where = np.where((min_x <= wl)*(max_x >= wl))
        elif isinstance(key, (np.ndarray, list, tuple)):
            where = []
            for k in key:
                where.append(self.__get_wl_idx__(k, wl))
        else:
            diff = (wl-key)**2
            where = np.argmin(diff)
                
        return np.array([where]).flatten().astype(int)
    
    def __get_idx_idx__(self, key, n_entry):
        idx = np.arange(n_entry)
        if isinstance(key, slice):
            where = idx[key]
        elif isinstance(key, (np.ndarray, list, tuple)):
            where = []
            for k in key:
                where.append(self.__get_idx_idx__(k, n_entry))
        else:
            diff = (idx-key)**2
            where = np.argmin(diff)
                
        return np.array([where]).flatten().astype(int)
    
    def __getitem__(self, key=None, **kwargs):
        if len(kwargs) != 0:
            print(kwargs)
        
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
        
        obj = self.__copy__()
        obj._reserved_attr = copy.deepcopy(self._reserved_attr)
        obj._custom_attr = copy.deepcopy(self._custom_attr)
        obj._pd_attr = copy.deepcopy(self._pd_attr)
        
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
        
        elif isinstance(key, (slice, list)):
            obj =  obj.iloc()[key,:]
        
        elif isinstance(key, str):
            keys = obj.get_labels()
            if key in keys:
                mi = obj.get_col_idx(key)
                obj =  pd.DataFrame.__getitem__(obj,mi)
                if key != 'spc':
                    obj.ga.is_spc = False
                    obj.ga.wavelength = None
                    obj.ga.unit_wl = None
                    obj.ga.n_wl = None
            else:
                warnings.warn('Item not found!')
                
        elif isinstance(key, tuple):
            if isinstance(key[0], str):
                if key[0] == 'wl':
                    wl_list = key[1:]
                    where = []
                    for k in wl_list:
                         where.append(obj.__get_wl_idx__(k, obj.ga.wavelength))
                         
                    where = np.concatenate(where)
                    where = np.unique(np.sort(where))
                    tmp = obj['spc', where]
                    obj.spc = tmp.spc.values
                    obj.ga.wavelength = tmp.ga.wavelength
                    obj.ga.n_wl = tmp.ga.n_wl
                elif len(key) >= 2:
                    col_idx = obj.get_col_idx(key[0])
                    n_entry = len(col_idx)
                    
                    where = []
                    for k in key[1:]:
                         where.append(obj.__get_idx_idx__(k,n_entry))
                         
                    where = np.concatenate(where)
                    where = np.unique(np.sort(where))
                    
                    tmp = pd.DataFrame.__getitem__(obj, col_idx[where])
                    
                    obj[key[0]] = tmp.values
                        
                    if key[0] == 'spc':
                        obj.ga.wavelength = obj.ga.wavelength[where]
                        obj.ga.n_wl = len(obj.ga.wavelength)
                else:
                    warnings.warn('key error')
            elif isinstance(key[0], (int,list,np.ndarray,slice,np.integer)):
                key = tuple(( 
                    k if not isinstance(k, (slice,int, np.integer)) 
                    else slice(k.start, k.stop-1, k.step) if isinstance(k, slice)
                    else slice(k, k, 1)
                    for k in key))
                obj = obj.loc[key,:]
            else:
                for i_key in key: # needs to be improved!
                    obj = obj.__getitem__(i_key)
            
        else:
            obj = pd.DataFrame.__getitem__(obj,key)
        obj.__update__()
        self.__update__()
        return obj
        
    def __delitem__(self, key):
        self.__update__()
        
        if key in self.get_labels():
            mi = self.get_col_idx(key)
            super(spc, self).__init__(self.drop(columns = mi))

        elif isinstance(key, (slice, int, np.integer)):
            key = self.index[key]
            super(spc, self).__init__(pd.DataFrame.drop(self, key))
        
        elif isinstance(key, MISlice):
            try:
                super(spc, self).__init__(pd.DataFrame.drop(self, key.slice_matrix))
            except:
                warnings.warn('Key not in rows')
        elif isinstance(key, pd.MultiIndex):
            try:
                super(spc, self).__init__(pd.DataFrame.drop(self, key))
            except:
                warnings.warn('Key not in rows')
                
        elif isinstance(key, tuple): #################
            if isinstance(key[0], str):
                if key[0] == 'wl':
                    wl_list = key[1:]
                    where = []
                    for k in wl_list:
                         where.append(self.__get_wl_idx__(k, self.ga.wavelength))
                         
                    where = np.concatenate(where)
                    where = np.unique(np.sort(where))
                    
                    mi = self.get_col_idx('spc')[where]
                    super(spc, self).__init__(pd.DataFrame.drop(self, mi, axis = 1))
                    self.ga.wavelength = np.delete(self.ga.wavelength, where)
                    self.ga.n_wl = len(self.ga.wavelength)
                    
                elif len(key) >= 2:
                    col_idx = self.get_col_idx(key[0])
                    n_entry = len(col_idx)
                    
                    where = []
                    for k in key[1:]:
                         where.append(self.__get_idx_idx__(k,n_entry))
                         
                    where = np.concatenate(where)
                    where = np.unique(np.sort(where))
                    
                    mi = self.get_col_idx(key[0])[where]
                    super(spc, self).__init__(pd.DataFrame.drop(self, mi, axis = 1))
                                            
                    if key[0] == 'spc':
                        self.ga.wavelength = np.delete(self.ga.wavelength, where)
                        self.ga.n_wl = len(self.ga.wavelength)
                        if self.ga.n_wl == 0:
                            self.ga.is_spc = False
                else:
                    warnings.warn('key error')
            elif isinstance(key[0], (int,list,np.ndarray,slice,np.integer)):
                key = tuple(( 
                    k if not isinstance(k, (slice,int, np.integer)) 
                    else slice(k.start, k.stop-1, k.step) if isinstance(k, slice)
                    else slice(k, k, 1)
                    for k in key))
                
                super(spc, self).__init__(pd.DataFrame.drop(self, self.loc[key,:].index, axis = 0))
            else:
                for i_key in key: # needs to be improved!
                    self = self.__delitem__(i_key)
            
        else:
            super(spc, self).__init__(pd.DataFrame.__delitem__(self, key))
        
        self.__update__()
        
        #### include multiindex for rows!
    """
    def __delitem__(self, key):
        self.__update__()
        
        if key in self.get_labels():
            mi = self.get_col_idx(key)
            super(spc, self).__init__(self.drop(columns = mi))

        elif isinstance(key, slice) or isinstance(key, int):
            #key = list(range(key.stop)[key])
            key = self.index[key]
            super(spc, self).__init__(pd.DataFrame.drop(self, key))
        else:
            super(spc, self).__init__(pd.DataFrame.__delitem__(self, key))
    """
    
    def __setitem__(self, key, value):
        
        if self.ga.is_empty == True:
                ## what happens if spc_data is a 1D list? --> each element is new spc by default
            if value.ndim == 1:
                value = np.reshape(value, (value.shape[0],1))
            ## other way arround: spcData = [spcData]
            
            data_shape = np.array(value.shape)[:-1]
            n_rows = np.prod(data_shape)
            
            self.ga.n_dim = len(data_shape)
            self.ga.data_shape = tuple(data_shape)
            
            wavelength = np.arange(value.shape[-1])
            
            if self.ga.n_dim > 1:
                index = [range(s) for s in self.ga.data_shape]
                index = pd.MultiIndex.from_product(index)
            else:
                index = np.arange(n_rows)
            
            # initialize an empty data frame
        
            if value.ndim == 1:
                k_key, key_list = self.__get_column_names(key, 1)
            else:
                k_key, key_list = self.__get_column_names(key, value.shape[-1])
            
            columns = pd.MultiIndex.from_product([[k_key],key_list])
            
            ra = self._reserved_attr
            ca = self._custom_attr
            pa = self._pd_attr
            
            super(spc, self).__init__(np.reshape(value, (n_rows, value.shape[-1]) ), index = index, columns = columns)
            self.index.names = range(len(self.index.names))
            
            self._reserved_attr = ra
            self._custom_attr = ca
            self._pd_attr = pa
            
            self.ga._reserved_attr['wavelength'] = wavelength
            self.ga._reserved_attr['n_wl'] = wavelength.size   
            self.ga.is_empty = False
        else:
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
        
        my_names = np.array(self.get_labels())
        rows = self.shape[0]
        my_names = my_names[my_names!='spc']
        if len(my_names) > 0:
            # first: look for the rows
            
            rowlim, rowex = self.__RowlimRowex()
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
                        my_label = list(map(_string_fun,values[i_col,:]))
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
    
    def __RowlimRowex(self):
        rowlim = self.OPTIONS['n_row_example_max']
        rowex = self.OPTIONS['n_row_example_pm']
            
        if 2*rowex > rowlim:
            rowex = np.amax([1, (np.ceil(rowlim/2)-1).astype(int)])
        return rowlim, rowex
    
    
    def short_label_repr(self, values):
        if not isinstance(values,(list, tuple, np.ndarray)):
            values = [values]
        def _string_fun(number):
            return self.__num_format(number, 3)
        rowlim, rowex = self.__RowlimRowex()
        rows = len(values)
        if rowlim == 1:
            r_list = list([0,])    
        elif rows > rowlim:
            r_list = list(np.concatenate([
                np.arange(rowex),
                np.arange(rowex)-rowex], axis = 0))
        
        else:
            r_list = list(np.arange(rows))
        
        my_label = list(map(_string_fun,np.array(values)[r_list]))
        
        
        if rowlim == 0:
            my_label = ''
        elif rowlim == 1:
            my_label = my_label[0]
        elif rows > rowlim :
            my_label = [(lambda i,k : k +', ' if i < 2*rowex else k)(i,k) for i,k in enumerate(my_label)]
            my_label.insert(rowex,'..., ')
            my_label = ''.join(my_label)[:-2]
        else:
            my_label = [(lambda i,k : k +', ' if i < rows else k)(i,k) for i,k in enumerate(my_label)]
            my_label = ''.join(my_label)[:-2]
            
        return my_label
                 
        
    def __show_attr(self):
        
        if len(self._custom_attr)>0:
            return ''.join([''.join(['  ',key,': ', str(self._custom_attr[key]), '\n']) for key in self._custom_attr])
        else:
            return '  none\n'
    
    def __str__(self):
        if self.ga.is_empty == True:
            return 'empty SpectralAnalysisPack object'
        if self.ga.is_spc:
            return 'SpectralAnalysisPack spc object, n-rows = %s' % (len(self)) 
        else:
            return 'SpectralAnalysisPack label object, n-rows = %s' % (len(self)) 
        
    
    
    def __repr__(self):
        self.__update__()
        
        if self.ga.is_empty == True:
            return 'empty SpectralAnalysisPack object'
        else:
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
            if self.ga.is_spc:
                return  ''.join([
                        'SpectralAnalysisPack spc object',
                        '\nnumber of rows: %s',
                        '\nnumber of channels: ', str(self.ga.n_wl),
                        '\nwavelength: ', self.short_label_repr(self.ga.wavelength),
                        '\nunit_wl: ', self.ga.unit_wl,
                        '\nspc: ', self.short_label_repr(self[0].spc.values.flatten()),
                        str_attr,
                        str_lab,
                        ]) % (self.shape[0]) 
            else:
                return  ''.join([
                        'SpectralAnalysisPack label object',
                        '\nnumber of rows: %s',
                        str_attr,
                        str_lab,
                        ]) % (self.shape[0]) 
        
        
    
    def get_col_idx(self, key):
        key_list = self.ga.columns.get_level_values
        key_list = key_list(1)[key_list(0) == key]
        key_list = pd.MultiIndex.from_product([[key], key_list])
        
        return key_list
    
    def __get_first_label__(self, only_name = False):
        labels = self.get_labels()
        if only_name == True:
            if 'spc' in labels:
                return 'spc'
            else:
                return labels[0]
        else:
            if 'spc' in labels:
                return self.__getitem__('spc').values
            else:
                return self.__getitem__(labels[0]).values
        
    def get_dim(self):
        index = self.index.__copy__().remove_unused_levels()
        level_range = [[min(l), max(l)] for l in index.levels]
        
        return level_range
    
    def shaped_array(self, filling = 0, filling_type = 'unnused', label = 'unnused'):
        level_range = np.array(self.get_dim())
        
        start = level_range[:,0]
        length = level_range[:,1]-start+1
                
        n_values = self.__get_first_label__().shape[-1]
                
        dims = np.concatenate([length, [n_values]])
        
        new_arr = np.ones(dims)*filling
        
        index = self.index.__copy__().remove_unused_levels().to_numpy()
        index = (np.array([list(x) for x in index]) - start).T
        
        new_arr[index[0],index[1]] = self.__get_first_label__()
        
        return new_arr
    
    # implementation of mathematical operations
    
    def __gt__(self, data):
        if hasattr(data, 'flat'):
            return self.values.flat > data.flat # .reshape(..., data.shape)???
        else:
            return self.values.flat > data
    
    def __ge__(self, data):
        if hasattr(data, 'flat'):
            return self.values.flat >= data.flat
        else:
            return self.values.flat >= data
    
    def __st__(self, data):
        if hasattr(data, 'flat'):
            return self.values.flat < data.flat
        else:
            return self.values.flat < data
    
    def __se__(self, data):
        if hasattr(data, 'flat'):
            return self.values.flat <= data.flat
        else:
            return self.values.flat <= data
        
    def __eq__(self, data):
        if hasattr(data, 'flat'):
            return self.values.flat == data.flat
        else:
            return self.values.flat == data

    def __ne__(self, data):
        if hasattr(data, 'flat'):
            return self.values.flat != data.flat
        else:
            return self.values.flat != data
        
    def __xor__(self, data):
        if hasattr(data, 'flat'):
            return self.values.flat ^ data.flat
        else:
            return self.values.flat ^ data