# SAVU_Py
  SAVU-Py (S)pectral (A)nalysis and (V)isualization (U)tilities in (Py)thon.
  A python package to load, oraganize, analyze and visualize hyperspectral data in Python.
  This work is based on the pandas dataframe.<br>

This is just a prototype. <br>
Example: reading a hyperspectral image from Witec confocal microscope:<br>

IN: my_spc = savu.read.from_Witec_head(fname, path)<br>
OUT: <br>
SpectralAnalysisPack spc object<br>
number of rows: 21025<br>
number of channels: 1024<br>
wavelength: 28.958, 33.783, 38.606, 43.426, ..., 3.882E+03, 3.885E+03, 3.888E+03, 3.891E+03<br>
unit_wl: rel. 1/cm<br>
spc: 626.000, 625.000, 623.000, 622.000, ..., 617.000, 622.000, 619.000, 622.000<br>
attributes:<br>
  xPix: 145<br>
  yPix: 145<br>
  channel: 1024<br>
  xDim: 10<br>
  yDim: 10<br>
  pixUnit: µm<br>
labels:<br>
  cluster: 0, 0, 3, 1, ..., 1, 1, 1, 1<br>


labels could be adressed by '.', e.g. <br>
  - my_spc.spc<br>
  - my_spc.cluster<br>
attributes need the .ga method:<br>
  - x = my_spc.ga.xPix<br>

