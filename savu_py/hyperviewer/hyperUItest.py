# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:32:00 2021

@author: basti
"""

from PyQt5 import QtCore, QtGui, QtWidgets, uic
import numpy as np
import os
import sys


class HyperViewer(QtWidgets.QMainWindow):
    
    def __init__(self, parent = None):
        super(HyperViewer, self).__init__(parent)
        
        filename = os.path.join(os.path.dirname(__file__), 'ui_files','ui_file_hyperviewer.ui')
        uic.loadUi(filename, self)
        
        self.show()
        

if __name__ == '__main__':      
    app = QtWidgets.QApplication(sys.argv)
    window = HyperViewer()
    app.exec_()