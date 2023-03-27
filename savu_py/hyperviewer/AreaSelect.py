# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:33:12 2020

@author: ungersebastian
"""
#%%
from PIL import Image
from PIL.ImageQt import ImageQt

from PyQt5.QtGui import QImage,QPixmap, QTransform, QPainter, QPen, QPolygon, QBrush, QColor
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFrame, QApplication
from PyQt5.QtCore import Qt, QPoint

import sys

import numpy as np

from shapely.geometry import Polygon, Point, LinearRing, LineString

import matplotlib

class AreaSelect(QGraphicsView):
    '''
    In the image, the area inside the polygonal countour line is selected.
    New points are added by either drag and drop or by a double click in the image
    Existing points are deleted by a double click near the point
    
    '''    
    DELTA = 10 #for the minimum distance
    COLOR =  [list(int(round(v*255)) for v in matplotlib.colors.to_rgb(c)) for c in matplotlib.pyplot.rcParams['axes.prop_cycle'].by_key()['color']]
    
    def __init__(self, image, parent = None, points = None):
        super(AreaSelect, self).__init__(parent)
        
        self.draggin_idx = -1  
        
        self._iArea = 0
        self._nAreaMax = len(self.COLOR)
        
        self._possibleAreas = list(range(self._nAreaMax))
        self._usedAreas = [self._possibleAreas.pop(self._iArea)]
        
        self._nArea = len(self._usedAreas)
        
        self._scene = QGraphicsScene(self)
        self._image = QGraphicsPixmapItem()
        self._scene.addItem(self._image)
        self.setScene(self._scene)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        
        self.image = image
        self.w, self.h = self.image.shape
        
        array = image.__copy__()
        
        """
        if len(array.shape) == 3:
            array = imProc.nip_to_color_image(array)
        """
        
        imgShow =  Image.fromarray(array)
        
        if imgShow.mode == 'F':
            imgShow = self.image_to_uint8(array)
        elif imgShow.mode == 'I;16':
            imgShow = self.image_to_uint8(array)
        
        
        qtImage1 = ImageQt(imgShow)
        qtImage2 = QImage(qtImage1)
        
        self.pixmap = QPixmap.fromImage(qtImage2)
        self.pixmap.detach()
        self.pixmapOrig = self.pixmap.copy()
        self.pixmapOrig.detach()
        
        self._image.setPixmap(self.pixmap)
        
        if points == None:
            self.points = [[
                    [self.h//2, self.w//3],
                    [self.h//3, 2*self.w//3],
                    [2*self.h//3, 2*self.w//3]
                    ]]
        else:
            self.points = points

        self.__create_new_plotmap__()
        self.win_scale = [1,1]
        self.height = self.w
        self.width = self.h
        
        self.show()
        
        self.win_scale = [1,1]
        self.area = np.zeros([self.w, self.h]).astype(int)
        self.__create_region__()
        self.__resize__()
        
    def __resize__(self):
        h = self.mapToScene(self.viewport().rect()).boundingRect().height()
        r = self.sceneRect()

        r.setHeight(h)
        self.setSceneRect(r)

        height = self.viewport().height()
        width = self.viewport().width()
        
        self.win_scale = [self.width / width, self.height / height]
        
        for item in self.items():
            item_height = item.boundingRect().height()
            item_width = item.boundingRect().width()
            tr = QTransform()
            tr.scale(width / item_width, height / item_height)
            item.setTransform(tr)
        
        
    def image_to_uint8(self, my_image):
        temp = (my_image - my_image.min())
        temp = temp / temp.max()
        temp = np.asarray(temp * 255, dtype=np.uint8)
        return( Image.fromarray(temp.astype(np.uint8), mode='L') )
            
       
    def __plot_points__(self):
        newMap = QPixmap(self.h, self.w)
        newMap.fill(Qt.transparent)
        
        painter = QPainter(self)
        painter.begin(newMap)
        for area, color in zip(self.points, self._usedAreas):
            pen = QPen(QColor(*self.COLOR[color]), 1)
            painter.setPen(pen)
            for p in area:
                painter.drawPoint(p[0] , p[1])
            pen = QPen(QColor(*self.COLOR[color]),0.5)
            painter.setPen(pen)
            
            brush = QBrush(QColor(*self.COLOR[color],20))
            painter.setBrush(brush)
            poly = QPolygon([QPoint(p[0], p[1]) for p in area])
            painter.drawPolygon(poly)
            
        painter.end()
            
        return(newMap)
    
    def __create_new_plotmap__(self):
        
        newMap = self.__plot_points__()
        
        newMapItem =  QGraphicsPixmapItem()
        newMapItem.setPixmap(newMap)
        
        r = self.sceneRect()        
        self._scene.addItem(newMapItem)
        self.setSceneRect(r)
        
    def __update_plotmap__(self):
        
        newMap = self.__plot_points__()
        
        newMapItem =  QGraphicsPixmapItem()
        newMapItem.setPixmap(newMap)
        
        items = self.items()
        self._scene.removeItem(items[0])
        del(items[0])
        self._scene.addItem(newMapItem)
        
        h = self.mapToScene(self.viewport().rect()).boundingRect().height()
        r = self.sceneRect()
        r.setHeight(h)
        self.setSceneRect(r)

        height = self.viewport().height()
        width = self.viewport().width()
        
        item_height = newMapItem.boundingRect().height()
        item_width = newMapItem.boundingRect().width()
        tr = QTransform()
        tr.scale(width / item_width, height / item_height)
        newMapItem.setTransform(tr)
    
    def __create_region__(self):
        pixmaps = [item.pixmap() for item in self.items()]
        ipm = 0
        image = pixmaps[ipm].toImage()
        b = image.bits()
        b.setsize(self.h * self.w * 4)
        arr = np.frombuffer(b, np.uint8).reshape((self.w, self.h, 4))
        arr = np.sum(arr, axis = 2)
        arr[arr>0]=1
        self.area = arr
        
    def resizeEvent(self, event):
        self.__resize__()       
        super(AreaSelect, self).resizeEvent(event)
    
    def _get_point(self, evt):
        point = np.array([evt.pos().x(),evt.pos().y()])*np.array(self.win_scale)
        if point[1] > self.height - 1:
            point[1] = self.height - 1
        if point[1] < 0:
            point[1] = 0
        if point[0] > self.width - 1:
            point[0] = self.width - 1
        if point[0] < 0:
            point[0] = 0
        return point
    
    def mouseDoubleClickEvent(self, evt):
        if evt.button() == Qt.RightButton and self._nAreaMax > self._nArea:
            newPoints = [
                    [self.h//2, self.w//3],
                    [self.h//3, 2*self.w//3],
                    [2*self.h//3, 2*self.w//3]
                    ]
            self.points.insert(self._iArea+1, newPoints)
            self._nArea=len(self.points)
            self._usedAreas.append(self._possibleAreas.pop(0))
            self._iArea += 1
            
            self.__update_plotmap__()
            
            
        if evt.button() == Qt.LeftButton and self.draggin_idx == -1:
            point = self._get_point(evt)
            dist = self.points[self._iArea] - point
            dist = np.sqrt(dist[:,0]**2 + dist[:,1]**2)
            if dist.min() > self.DELTA:
                poly = Polygon(self.points[self._iArea])
                gpoint = Point(point[0], point[1])
                pol_ext = LinearRing(poly.exterior.coords)
                d = pol_ext.project(gpoint)
                p = pol_ext.interpolate(d)
                closest_point_coords = list(p.coords)[0]
                gpoint = Point(closest_point_coords)
        
                n = len(self.points[self._iArea])
                p = [(k, k+1) for k in np.arange(n-1)]
                p.append((n-1,0))
                
                k = np.argmin([LineString([self.points[self._iArea][line[0]], self.points[self._iArea][line[1]]]).distance(gpoint) for line in p])
                
                coords = p[k]
                
                if max(coords) == n-1 and min(coords) == 0:
                    insert = 0
                else:
                    insert = max(coords)
                
                self.points[self._iArea].insert(insert, point)
                
                self.__update_plotmap__()
                self.__create_region__()
            elif dist.min() <= self.DELTA and len(self.points[self._iArea])>3:
                self.points[self._iArea].pop(dist.argmin())
                self.__update_plotmap__()
                self.__create_region__()
                
    def mousePressEvent(self, evt):
        if evt.button() == Qt.RightButton:
            if self._iArea == self._nArea-1:
                self._iArea = 0
            else:
                self._iArea += 1  
        elif evt.button() == Qt.LeftButton and self.draggin_idx == -1:
            point = self._get_point(evt)
            dist = self.points[self._iArea] - point
            dist = np.sqrt(dist[:,0]**2 + dist[:,1]**2)
            dist[dist>self.DELTA] = np.inf
            if dist.min() < np.inf:
                self.draggin_idx = dist.argmin()        

    def mouseMoveEvent(self, evt):
        if self.draggin_idx != -1:
            point = self._get_point(evt)
            self.points[self._iArea][self.draggin_idx] = point
            self.__update_plotmap__()

    def mouseReleaseEvent(self, evt):
        if evt.button() == Qt.LeftButton and self.draggin_idx != -1:
            point = self._get_point(evt)
            self.points[self._iArea][self.draggin_idx] = point
            self.draggin_idx = -1
            self.__update_plotmap__()
            self.__create_region__()
    
    def closeEvent(self,evt):
        QApplication.quit()

def getArea(image, points = None, delta = 10):
    app = QApplication(sys.argv)
    ex = AreaSelect(image = image, points = points)
    ex.DELTA = delta
    app.exec_()
    
    my_p=ex.points
    my_a=ex.area
    
    return my_p, my_a

#%% Example 1: select Area on example image
if __name__ == '__main__':
    import NanoImagingPack as nip
    my_im = nip.readim('lena')[:,50:]
    my_p, my_a = getArea(my_im)
#%% Example 2: use previously obtained points
if __name__ == '__main__':
    my_p, my_a = getArea(my_im, my_p)

