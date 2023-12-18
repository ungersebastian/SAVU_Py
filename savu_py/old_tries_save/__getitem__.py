# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:51:25 2023

@author: basti
"""

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