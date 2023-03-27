# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:44:38 2021

@author: basti
"""

from os.path import dirname, isfile, join, splitext, basename
from os import listdir
import numpy as np

from ..spc import spc

def anfatec(path = None, *args, **kwargs):
    _data_type_  = np.dtype(np.int32)
    
    # name of data
    name = splitext(basename(path))[0]
    
    # get file directory
    directory = dirname(path)
    
    # list all files in directory
    file_list = np.array([(
        f,                  # filename
        join(directory, f), # absolute path
        splitext(f)[1]      # filename without extention
        ) for f in listdir(directory) if isfile(join(directory, f))])
    
    # open the headerfile
    with open(path, 'r') as fopen:
        header_list = np.array(fopen.readlines())
        
    # extract the file list and supporting information
    
    where = np.where([('FileDesc' in hl) and ('Begin\n' in hl) for hl in header_list])[0]
    where = np.append(where, len(header_list))
    files = [header_list[where[i]:where[i+1]] for i in range(len(where)-1)]
    files[-1] = files[-1][0:np.where([('FileDesc' in hl) and ('End\n' in hl) for hl in header_list])[0][0]+1]
    files = [f[( np.where([('FileDesc' in hl) and ('Begin\n' in hl) for hl in f])[0][0]+1):( np.where([('FileDesc' in hl) and ('End\n' in hl) for hl in f])[0][0])] for f in files]
        
    header_list = header_list[0:where[0]-1]
    
    del(where)
    
    anfatec_dict = __init_dict__(header_list)
    
    del(header_list)
    
    files = [__return_dict__(f) for f in files]
    files = [f for f in files if f !=  {}]
    
    # get the wavelength axis and the other main informations
    fCount = 0
    for f in files:
        if 'FileNameWavelengths' in f and 'PhysUnitWavelengths' in f:
            break
        else:
            fCount = fCount+1
    spc_file = files.pop(fCount)
    
    # get the wavelength axis
    path_wavelengths = spc_file.pop('FileNameWavelengths')
    path_wavelengths = file_list[file_list[:,0]==path_wavelengths][0,1]
    
    with open(path_wavelengths, 'r') as fopen:
        wavelength = fopen.readlines()
        
    wavelength = [''.join(l.split('\n')) for l in wavelength][1:]
    wavelength = (np.array([l.split('\t') for l in wavelength]).T).astype(float)
    
    anfatec_dict['wavelength'] = wavelength[0]
    anfatec_dict['attenuation'] = wavelength[1]
    
    for kFile in spc_file.keys():
        anfatec_dict[kFile] = spc_file[kFile]

    del(wavelength, path_wavelengths)
    
    # extract the file information

    newfile ={
     'FileName' : anfatec_dict.pop('FileName'),
     'Caption' : anfatec_dict.pop('Caption'),
     'Scale' : anfatec_dict.pop('Scale'),
     'PhysUnit' : anfatec_dict.pop('PhysUnit')
     }
    
    files.append(newfile)
    anfatec_dict['files'] = files
    
    del(files, newfile)
    
    # read the images
        
    for my_file in anfatec_dict['files']:
        path_file = join(directory, my_file['FileName'])
        with open(path_file, 'rb') as fopen:
            my_im = fopen.read()
            my_im = np.frombuffer(my_im, _data_type_)
            my_dim = int(len(my_im)/(anfatec_dict['xPixel']*anfatec_dict['yPixel']))
            if my_dim == 1:
                news = (anfatec_dict['xPixel'], anfatec_dict['yPixel'])
            else:
                news = (anfatec_dict['xPixel'], anfatec_dict['yPixel'], my_dim)
            my_im = np.reshape(my_im, news)
            
        scale = my_file['Scale']
        my_file['data'] = my_im*scale
        
    # create spc object
    spc_data = anfatec_dict['files'][-1]
    spc_anfatec = spc(
        spc = spc_data['data'],
        unit_wl = anfatec_dict.pop('PhysUnitWavelengths'),
        unit_spc = spc_data['PhysUnit'],
        wavelength = anfatec_dict['wavelength']
        )
    # Caption
    return spc_anfatec, anfatec_dict

def __return_value__(v):
    """

    Parameters
    ----------
    v : str
        Value to be convertet.

    Returns
    -------
    int or float type of value.

    """
    try:
        v = int(v)
    except:
        try:
            v = float(v)
        except:
            pass
    return v

def __init_dict__(arr):
    """

    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    anfatec_dict = dict()
    arr = arr[[':' in l for l in arr]]
    arr = [''.join(l.split()) for l in arr]
    arr = [''.join(l.split('\n')) for l in arr]
    
    for l in arr:
        anfatec_dict[l.split(':', 1)[0]] = __return_value__(l.split(':', 1)[1])
    

    
    return anfatec_dict

def __cleanData__(s):
    s = s.split(':')
    s0 = s[0]
    s1 = s[1]
    if ''.join(s0.split()) == 'FileName':
        if len(s1.split()) > 1:
            return False
    return True

def __return_dict__(arr):
    arr = arr[[':' in l for l in arr]]
    
    check = [__cleanData__(l) for l in arr]
    if False in check:
        return {}
    else:
        arr = [''.join(l.split()) for l in arr]
        arr = [''.join(l.split('\n')) for l in arr]
        return {l.split(':', 1)[0]: __return_value__(l.split(':', 1)[1]) for l in arr}

