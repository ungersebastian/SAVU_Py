# SAVU_Py
  SAVU-Py (S)pectral (A)nalysis and (V)isualization (U)tilities in (Py)thon.
  A python package to load, oraganize, analyze and visualize hyperspectral data in Python.
  This work is based on the pandas dataframe.

This is just a prototype. 
Example: reading a hyperspectral image from Witec confocal microscope:

IN: my_spc = savu.read.from_Witec_head(fname, path)
OUT: 
SpectralAnalysisPack spc object
number of rows: 21025
number of channels: 1024
wavelength: 28.958, 33.783, 38.606, 43.426, ..., 3.882E+03, 3.885E+03, 3.888E+03, 3.891E+03
unit_wl: rel. 1/cm
spc: 626.000, 625.000, 623.000, 622.000, ..., 617.000, 622.000, 619.000, 622.000
attributes:
  xPix: 145
  yPix: 145
  channel: 1024
  xDim: 10
  yDim: 10
  pixUnit: µm
labels:
  cluster: 0, 0, 3, 1, ..., 1, 1, 1, 1


labels could be adressed by '.', e.g. 
  - my_spc.spc
  - my_spc.cluster
attributes need the .ga method:
  - x = my_spc.ga.xPix

