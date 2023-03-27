import numpy as np
import pandas as pd
import copy    
import warnings

#from label import label as lb

class spc(pd.DataFrame):
    _metadata = ['custom_attr','reserved_attr']

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
            super(pd.DataFrame, self).__init__(args[0])
        else:
            # dict for custom attributes
            self.custom_attr = dict()
            
            # dict for reserved attributes
            self.reserved_attr = {
                    'wavelength': None,                         
                    'unit': None,                               
                    'n_spc': None,                              
                    'shape_spc': None,                          
                    'n_wl': None,                               
                    'columns': {'unnamed': np.array([])}
                    }
            
            # check if this is for label subclass!
            
            if 'is_label' in kwargs:
                is_label = kwargs.pop('is_label')
            else:
                is_label = False
            
            if not is_label:
                # unpacking of args and kwargs:
                i_args = 0
                
                # setting the spc attribute
                if 'spc' in kwargs:
                    spcData = np.array(kwargs.pop('spc', None))
                elif len(args)-i_args > 0:
                    spcData = np.array(args[i_args])
                    i_args += 1
                else:
                    raise ValueError('spc must be given!')
                
                self.shape_spc = np.array(spcData.shape)[:-1]
                self.n_spc = np.prod(self.shape_spc)
    
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
                
                self.wavelength = wavelength
                self.n_wl = wavelength.size
                
                # setting the unit attribute
                if 'unit' in kwargs:
                    unit = str(kwargs.pop('unit', None))
                elif len(args)-i_args > 0:
                    unit = str(args[i_args])
                    i_args += 1
                else:
                    unit = ''
                
                self.unit = unit
                
                 # initializing
                super(spc, self).__init__(
                        spcData,
                        columns = self.__set_column_names('spc', self.n_wl)
                        )
            else:
                #create empty dataframe
                i_args = 0
                my_empy_frame = pd.DataFrame()
                print('Im here!')
                super(spc, self).__init__(my_empy_frame)
                super(pd.DataFrame, self).__init__(data = [])
            
            # adding of labels
            for key in kwargs:
                self.add_label(key, kwargs[key])
            
            i_label = 0
            while i_args+i_label < len(args):
                #key = ''.join(['unnamed_', str(i_label+1)])
                key = 'unnamed'
                self.add_label(key, args[i_args+i_label])
                i_label += 1
    
    def __set_column_names(self, name, length):
        if name == 'unnamed':
            n_unnamed = self.reserved_attr['columns']['unnamed'].size
            if n_unnamed == 0:
                n_index = 1
            else:
                new_ind = np.array(range(n_unnamed + 1))+1
                old_ind = self.reserved_attr['columns']['unnamed']
                n_index = new_ind[np.where(np.array([ind in old_ind for ind in new_ind]) == False)[0][0]]
            name = ''.join(['unnamed_',str(n_index)])
            self.reserved_attr['columns']['unnamed'] = np.append(self.reserved_attr['columns']['unnamed'], n_index).astype(int)

        
        if length == 1:
            my_column = [name]
        else:
            my_column = [''.join([name, '_', str(col_nr+1)]) for col_nr in range(length)]
        

        self.reserved_attr['columns'][name] = my_column
        return my_column
    
    def __getattr__(self, name):
        if name.startswith('_'):
            return pd.DataFrame.__getattr__(self, name)
        elif name in self._metadata:
            return pd.DataFrame.__getattr__(self, name)
        elif name in self.reserved_attr:
            return self.reserved_attr[name]
        elif name in self.custom_attr:
            return self.custom_attr[name]
        elif name in self.reserved_attr['columns']:
            key_list = self.reserved_attr['columns'][name]
            val = pd.DataFrame.__getitem__(self, key = key_list).values
            if np.array(self.reserved_attr['columns'][name]).size == 1:
                return np.ravel(val)
            else:
                return val
        else:
            return pd.DataFrame.__getattr__(self, name)
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            pd.DataFrame.__setattr__(self, name, value)
        elif name in self._metadata:
            pd.DataFrame.__setattr__(self, name, value)
        elif name in self.reserved_attr:
            self.reserved_attr[name] = value
        else:
            self.custom_attr[name] = value
            
    def __hasattr__(self, name):
        if name.startswith('_'):
            return pd.DataFrame.__hasattr__(self, name)
        elif name in self._metadata:
            return pd.DataFrame.__hasattr__(self, name)
        elif name in self.reserved_attr:
            return True
        elif name in self.reserved_attr['columns']:
            return True
        elif name in self.custom_attr:
            return True
    
    def __delattr__(self, name):
        if name.startswith('_'):
            print('Attribute can´t be removed!')
            return None
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
                obj.custom_attr = copy.deepcopy(self.custom_attr)
                obj.reserved_attr = copy.deepcopy(self.reserved_attr)
                
                obj.spc = new_spc.__copy__()
                obj.wavelength = new_wl.__copy__()
                return obj
                
        else:
            obj = copy.deepcopy(self)
            obj.custom_attr = copy.deepcopy(self.custom_attr)
            obj.reserved_attr = copy.deepcopy(self.reserved_attr)
            
        if isinstance(key, int):
            new_spc = obj.spc[key,:]
            
            obj = pd.DataFrame.__getitem__(obj,slice(key,key+1))
            obj.custom_attr = copy.deepcopy(self.custom_attr)
            obj.reserved_attr = copy.deepcopy(self.reserved_attr)
            obj.spc = new_spc
            obj.n_spc = 1
            obj.shape_spc = np.array([1])
            
            return obj   
        elif not isinstance(key, slice):
            if key in self.reserved_attr['columns']:
                key_list = self.reserved_attr['columns'][key]
                return pd.DataFrame.__getitem__(self, key = key_list)
            elif key in self.columns:
                return pd.DataFrame.__getitem__(self, key = key)
        
        else:
            return pd.DataFrame.__getitem__(self,key)
    def __setitem__(self, key, value):
        if key in self.reserved_attr['columns']:
            key_list = self.reserved_attr['columns'][key]
            val_shape = np.array(value).shape
            if len(val_shape) == 2:
                if val_shape[0] == self.n_spc and val_shape[1] == len(key_list):
                    pd.DataFrame.__setitem__(self, key = key_list, value = value)
                else:
                    warnings.warn('Dimension of value did not fit!')
            elif len(val_shape) == 1 and len(key_list) == 1:
                pd.DataFrame.__setitem__(self, key, value)
            else:
                warnings.warn('Dimension of value did not fit!')
        elif key in self.columns:
            pd.DataFrame.__setitem__(self, key = key, value = value)
            
    def __delitem__(self, key):
        
        if key in self.reserved_attr['columns']:
            key_list = self.reserved_attr['columns'][key]
            pd.DataFrame.__delitem__(self, key = key_list)
            del self.reserved_attr['columns'][key]
        elif key in self.columns:
             pd.DataFrame.__delitem__(self, key = key)
        else:
            warnings.warn('Key not found!')
        

            
            
        
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
        my_names = list(self.reserved_attr['columns'].keys())
        my_names.remove('spc')
        my_names.remove('unnamed')
        if len(my_names) > 0:
            my_str = np.array([])
            for name in my_names:
                my_list = self.reserved_attr['columns'][name]
                if len(my_list) > 1:
                    print(name)
                    my_str = np.append(my_str, [''.join(['  ',name, ':\n'])])
                    my_str = np.append(my_str, [''.join([''.join(['    ',str(ind+1), ': ', self.__show_label(self[val]), '\n']) for ind, val in enumerate(my_list)])])
                else:
                    my_str = np.append(my_str, [''.join(['  ',name, ': ', self.__show_label(self[name]), '\n'])])
            my_str = ''.join(my_str)
        else:
            my_str = '  none\n'

        return my_str
    
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
    
    def add_label(self, key, val):
        val = np.array(val)
        if val.ndim == 0:
            val = np.full(self.n_spc, val)
        elif val.size == 1:
            val = np.full(self.n_spc, np.ravel(val)[0])
        elif (val.ndim == 1 and val.size != self.n_spc) or (val.ndim == 2 and val.shape[0] == 1):
            val = np.ravel(val)
            val = np.full((self.n_spc, val.size), val)
        elif val.ndim >= 2:
            if val.ndim == 2 and val.shape[0] == self.n_spc:
                pass
            else:
                raise ValueError('Dimensionality of label too high!')
        
        if val.ndim == 1:
            key_list = self.__set_column_names(key, 1)
        else:
            key_list = self.__set_column_names(key, val.shape[1])
        
        new_frame = pd.DataFrame(val, columns = key_list)
        super(spc, self).__init__(pd.concat([self, new_frame], axis=1, sort=False))
            
            

"""             
            
            
            
            
                
            
            

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #def __setitem__(self,key, val):
    #    pd.DataFrame.__setitem__(self, key, val)
    
   
        
    
""" 