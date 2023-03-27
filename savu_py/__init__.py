#!/usr/bin/env python3<
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 00:03:21 2017

@author: root
"""

# externals
import numpy as np;

# modules
from .spc import *;
from .MISlice import *;
from .utils import read;

# reading functions
from .read import anfatec;
from .read import witec;
from .read.witec import *;

# cluster functions
from .cluster import kmeans;
from .decomp import pca;

from .__get_attribute__ import __get_attribute__
from .utils import __predict__
#from .resources import *
#import read;

## packages
from . import resources;

