#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 23:38:28 2018

@author: unger
"""

import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    # ext_mdoules = [External_code/cos_module],        <- for incorporating c-files
    name="SAVU_Py",
    version="0.0.0dev1",
    author="Sebastian Unger",
    author_email="basti.unger@googlemail.com",
    description="A package for spectral analsis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    
    include_package_data=True,
    package_data={'SAVU_Py':['resources/*']},
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
)
