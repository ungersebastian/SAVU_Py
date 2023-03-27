import numpy as np
import pandas as pd
import copy    

class spc(pd.DataFrame):
    _metadata = ['custom_attr','reserved_attr']
    custom_attr = dict()
    reserved_attr = dict()

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
        if name in self.reserved_attr:
            self.reserved_attr[name] = value
        elif name in self.custom_attr:
            self.custom_attr[name] = value
        else:
            pd.DataFrame.__setattr__(self, name, value)
    
    def __getattr__(self, name):
        if name in self._metadata:
            return pd.DataFrame.__getattr__(self, name)
        elif name in self.reserved_attr:
            return self.reserved_attr[name]
        elif name in self.custom_attr:
            return self.custom_attr[name]
    
    def __hasattr__(self, name):
        if name in self._metadata:
            return pd.DataFrame.__hasattr__(self, name)
        elif name in self.reserved_attr:
            return True
        elif name in self.custom_attr:
            return True
    
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
 