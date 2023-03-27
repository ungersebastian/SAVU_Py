# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:08:10 2019

@author: ungersebastian
"""


import mne
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the internet
path = mne.datasets.kiloword.data_path() + '/kword_metadata-epo.fif'
epochs = mne.read_epochs(path)

# The metadata exists as a Pandas DataFrame
print(epochs.metadata.head(10))