# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 13:06:31 2021

@author: basti
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:21:42 2019

@author: Nancy
"""
import os
import warnings
import numpy as np
import savu_py as sap

def from_Witec_head(fname, path = os.getcwd(), kind='scan'):
    f = open(path + '/' + fname);
    f = f.readlines();
    
    head_read = dict(filter(lambda x : len(x) > 1 , map(lambda y : y.split(' = '), f)))
    paraWitec  = read_Spc(head_read, kind)
    print(paraWitec)
    xValues = np.loadtxt(path + '/' + paraWitec.pop('xValues'))
    yValues = np.loadtxt(path + '/' + paraWitec.pop('yValues'))
    
    yValues = np.reshape(yValues, (paraWitec['xPix'], paraWitec['yPix'], paraWitec['channal'] ))
    
    paraWitec['pixUnit'] = paraWitec.pop('unit')
    spc = sap.spc(
        spc = yValues, 
        wavelength = xValues,
        unit = paraWitec.pop('xUnit')
        )
    for key in paraWitec.keys():
        spc.ga(key, paraWitec[key])
        
    return spc
    
   

def import_spc (values, path=os.getcwd()):
    #print(kind)
    
    # x-values for the wavelength
    xValues = np.loadtxt(path + '/' + values['xValues'])
       
    # y-values for the spectral intensties
    yValues = np.loadtxt(path + '/' + values['yValues']);
    yValues = np.reshape(yValues, (values['xPix'], values['yPix'], values['channal'] ))
        
    return (xValues, yValues)

# function for reading Witec_headers
def header_Witec(filename, path=os.getcwd(), kind='no scan'):
    '''
    Reading of header-textfiles from Witec export
    params:
        expect
            path     ... a string (standard cwd)
            kind     ... a string ('scan' or standard 'no scan')
            filename ... a string
        return
            opened file as dict
    '''
    f = open(path + '/' + filename);
    f = f.readlines();
    
    # split of lines to get key and value pairs (map(lambda y...))
    # filter to remove list elements that containing only keys (filter(lambda x...))
    # convertion of internal list to dict
    
    head_read = dict(filter(lambda x : len(x) > 1 , map(lambda y : y.split(' = '), f)))
    paraWitec = read_Spc(head_read, kind)
    return (paraWitec, path)

# function for reading the information of Witec_headers
def read_Spc(Witec_head, kind):
    '''
    Reading of spectral data from Witec export with header file
    params:
        expect
            Witec_head ... dictionary of keys and values extrakted from Witec header
            kind       ... remand from header_Witec: a string ('scan' or standard 'no scan')
                           'no scan' only x- and yValues and the xUnit are used even if the exported file is an image scan
        return
            values     ... list of values
    '''
    values={};
    if 'GraphName' in Witec_head :
        xValues = Witec_head['GraphName'].split('\n')[0] + ' (X-Axis).txt'
        yValues = Witec_head['GraphName'].split('\n')[0] + ' (Y-Axis).txt'
        values["xValues"] = xValues
        values["yValues"] = yValues
    else:
        raise ValueError('No such string: GraphName')
    
    if 'XAxisUnit' in Witec_head :
        xUnit = Witec_head['XAxisUnit'].split('\n')[0]
        values["xUnit"] = xUnit
    else:
        raise ValueError('No such string: XAxisUnit')
    
    if kind == 'scan':
        scan_Witec(Witec_head, values)
    else:
        spc_Witec(Witec_head, values)
                  
    return values

# function for reading Witec_headers of image scans
def scan_Witec(Witec_head, values):
    '''
    Reading only spectral date from Witec export of image scans with header files 
    params:
        expect
            Witec_head ... dictionary of keys and values extrakted vom Witec header prefiled from read_Spc
            values     ... list of values prefiled by read_Spc
        return
            values     ... list of values (from read_Spc and scan params)
    '''    
    # Parameters for determination of map-demention and pixelsize
    # map-dimension number of pixels
    if 'SizeX' in Witec_head :
        xPix = Witec_head['SizeX'].split('\n')[0]
        values["xPix"] = int(xPix)
    else:
        raise ValueError('No such string: SizeX')
           
    if 'SizeY' in Witec_head :
        yPix = Witec_head['SizeY'].split('\n')[0]
        values["yPix"] = int(yPix)
    else:
        raise ValueError('No such string: SizeY')
        
    # number of wavelength chanals
    if 'SizeGraph' in Witec_head :
        channal = Witec_head['SizeGraph'].split('\n')[0]
        values["channal"] = int(channal)
    else:
        raise ValueError('No such string: SizeY')
              
    # Axsis-dimension
    if 'ScanWidth' in Witec_head :
        xSize = Witec_head['ScanWidth'].split('\n')[0]
        values["xDim"] = int(xSize)
    else:
        raise ValueError('No such string: ScanWidth')
           
    if 'ScanHeight' in Witec_head :
        ySize = Witec_head['ScanHeight'].split('\n')[0]
        values["yDim"] = int(ySize)
    else:
        raise ValueError('No such string: ScanHeight')
          
    if 'ScanUnit' in Witec_head :
        unit = Witec_head['ScanUnit'].split('\n')[0]
        values["unit"] = unit
    else:
        raise ValueError('No such string: ScanUnit')
    
    return values

# funtion for reading the information of Witec_heasders that aren't images scans
def spc_Witec(Witec_head, values):
    '''
    Reading only spectral data from Witec export of single or series spectral data with header files 
    params:
        expect
            Witec_head ... dictionary of keys and values extrakted vom Witec header prefiled from read_Spc
            values     ... list of values prefiled by read_Spc
        return
            values     ... list of values (from read_Spc only or with values of series data)
    '''    
    #if 'SizeX' && 'SizeY' in Witec_head && Witec_head['SizeX'].split('\n')[0] >= 1 && Witec_head['SizeX'].split('\n')[0] != Witec_head['SizeY'].split('\n')[0]:
    if 'SeriesUnit' in Witec_head:
        fast_Witec(Witec_head, values)
    else:
        return values

# funktion of reading information from Witec_headers of series
def fast_Witec (Witec_head, values):
    if 'SizeX' in Witec_head :
        xPix = Witec_head['SizeX'].split('\n')[0]
        values["xPix"] = xPix
    else:
        raise ValueError('No such string: SizeX')
        
    return values

