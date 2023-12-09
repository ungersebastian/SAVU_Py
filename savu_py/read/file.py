# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 21:30:12 2023

@author: basti
"""

def read_csv(filename, chunksize = 10**5):
    
    chunks = pd.read_csv(
        filename,
        chunksize = chunksize,
        header = None,
        sep=',',
        escapechar='\\'
        )
    
    df = pd.concat(chunks)
    
    return df.values