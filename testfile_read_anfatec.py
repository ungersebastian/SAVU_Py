# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:50:32 2021

@author: basti
"""

import savu_py as savu
import os
import matplotlib.pyplot as plt
import numpy as np

path_final = r'C:\Users\basti\Python_Projekte\irafmimaging\NanIRspec\resources\2107_weakDataTypeJuly'
headerfile = r'BacVan30_0013.txt'

fName = os.path.join(path_final, headerfile)

spc, d = savu.read.anfatec(fName) 

"""
del spc[17,19:26]
del spc[17,7]
del spc[21,8]
"""

"""
del spc[0:32,0]
del spc[25,25]
"""

val = savu.utils.norm(spc.spc, 2 , apply = True).spc.values
#val = spc.spc.values
spc.id = np.arange(len(val))

sid = spc.id.values.flatten()

img = spc.shaped_array()

plt.figure()
plt.imshow(np.sum(img, axis = -1))
plt.show()

n_neighbors = 10
from sklearn.neighbors import NearestNeighbors

def metric(s1, s2):
    v1 = np.std(s1)
    v2 = np.std(s2)
    if v1>0 and v2>0:
        st1 = (s1-np.mean(s1))/v1
        st2 = (s1-np.mean(s2))/v2
        div = np.sqrt(np.sum(st1**2)*np.sum(st2**2))
        if div == 0:
            return 0
        else:
            return np.arccos(np.sum(st1*st2)/div)
    else:
        return 0

nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(val)
#nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree', metric = metric).fit(val)
distances, indices = nbrs.kneighbors(val)
dist2 = 1/(distances+np.mean(distances)*1E-3)
p = sid
indices = indices
distances = distances
import networkx as nx
G = nx.Graph()
G.add_nodes_from(p)

[[G.add_edge(node, ind, length = edge_len) for ind, edge_len in zip(node_ind, node_edge)] for node, node_ind, node_edge in zip(p,indices,dist2)]
#[[G.add_edge(node, ind, length = edge_len) for ind, edge_len in zip(node_ind, node_edge)] for node, node_ind, node_edge in zip(p,indices,distances)]
#%%
pos = nx.spring_layout(G, weight='length')
#pos = nx.kamada_kawai_layout(G)

plt.figure()
#subax1 = plt.subplot(121)
nx.draw(G, pos, node_size = 1)
#subax2 = plt.subplot(122)
#nx.draw(G, nx.spring_layout(G), node_size = 1)
plt.show()

#%%
np.save('posTest',np.array(list(pos.values())))

#%%
img = spc.shaped_array()

plt.figure()
plt.imshow(np.sum(img, axis = -1))
plt.show()

plt.figure()
#subax1 = plt.subplot(121)
nx.draw(G, pos, node_size = 1)
#subax2 = plt.subplot(122)
#nx.draw(G, nx.spring_layout(G), node_size = 1)
plt.show()
#%%
positions = np.load('posTest.npy').T


ax = 0.45
tmp2 = positions[:,positions[0] < ax]
tmp1 = positions[:,positions[0] >= ax]
pRest = tmp2

ay = 0.45
tmp2 = tmp1[:,tmp1[1] < ax]
tmp1 = tmp1[:,tmp1[1] >= ay]

c1 = tmp1
pRest = np.concatenate((pRest, tmp2), axis = -1)

ax = -0.05
tmp2 = pRest[:,pRest[0] < ax]
tmp1 = pRest[:,pRest[0] >= ax]
pRest = tmp2

ay = -0.07
tmp2 = tmp1[:,tmp1[1] < ax]
tmp1 = tmp1[:,tmp1[1] >= ay]

c2 = tmp1
pRest = np.concatenate((pRest, tmp2), axis = -1)

ax = -0.87
tmp2 = pRest[:,pRest[0] >= ax]
tmp1 = pRest[:,pRest[0] < ax]
pRest = tmp2
c3 = tmp1

aa = 0.25
ab = -1.6
tmp2 = pRest[:,pRest[0]+ab*pRest[1] >= aa]
tmp1 = pRest[:,pRest[0]+ab*pRest[1] < aa]
pRest = tmp2
c4 = tmp1

ay = -0.6
tmp2 = pRest[:,pRest[1] >= ay]
tmp1 = pRest[:,pRest[1] < ay]
c5 = tmp1
c6 = tmp2




p = positions.T
c = [c1.T , c2.T, c3.T, c4.T, c5.T, c6.T]
cid = [[np.where(p == ic)[0][0] for ic in ica] for ica in c]
classes = np.zeros(len(p))-1
for iv, ica in enumerate(cid):
    classes[ica] = iv
    
import matplotlib as mpl

cmap = plt.cm.jet  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
cmaplist[0] = (.5, .5, .5, 1.0)

# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

plt.figure()
plt.scatter(c1[0], c1[1], color = cmaplist[256//5*0])
plt.scatter(c2[0], c2[1], color = cmaplist[256//5*1])
plt.scatter(c3[0], c3[1], color = cmaplist[256//5*2])
plt.scatter(c4[0], c4[1], color = cmaplist[256//5*3])
plt.scatter(c5[0], c5[1], color = cmaplist[256//5*4])
plt.scatter(c6[0], c6[1], color = cmaplist[256//5*5])
plt.show()

spc.man = classes

img = spc.man.shaped_array()

plt.figure()
plt.imshow(img, cmap=cmap)
plt.show()

#%%

s = [np.mean(val[classes == ic], axis = 0) for ic in range(6)]
plt.figure()
for ii, ic in enumerate(s):
    plt.plot(ic, color = cmaplist[256//5*ii])
plt.show()

#%%

cluster = savu.cluster.kmeans(savu.utils.norm(spc.spc, 2 , apply = True),6)

img = cluster.shaped_array()
plt.figure()
plt.imshow(img, cmap=cmap)
plt.show()

cluster = savu.cluster.kmeans(spc,6)

img = cluster.shaped_array()
plt.figure()
plt.imshow(img, cmap=cmap)
plt.show()
#%%
lensSubset = np.zeros(len(p))-1

pValArray = np.array(list(pos.values()))
pPosArray = np.array(list(pos.keys()))


nSubEdge = 10
nOver = 0


minX, maxX = np.amin(pValArray[:,0]), np.amax(pValArray[:,0])
minY, maxY = np.amin(pValArray[:,1]), np.amax(pValArray[:,1])
dX = (maxX - minX)/nSubEdge
dY = (maxY - minY)/nSubEdge

xList = [[minX+nEdge*dX, minX+(nEdge+1)*dX] for nEdge in range(nSubEdge)]
yList = [[minY+nEdge*dY, minY+(nEdge+1)*dY] for nEdge in range(nSubEdge)]


for kx in range(nSubEdge):
    for ky in range(nSubEdge): 
        sel = (pValArray[:,0] >= xList[kx][0]) * (pValArray[:,0] <= xList[kx][1]) * (pValArray[:,1] >= yList[ky][0]) * (pValArray[:,1] <= yList[ky][1])
        lensSubset[pPosArray[sel]] = kx*nSubEdge+ky

spc.lens = lensSubset

img = spc.lens.shaped_array()


import matplotlib as mpl

cmap = plt.cm.jet  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
cmaplist[0] = (.5, .5, .5, 1.0)

# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

plt.figure()
plt.imshow(img, cmap=cmap)
plt.show()

#%%
# now: overlapping subsets

nSubEdge = 3
nOver = 0.2

minX, maxX = np.amin(pValArray[:,0]), np.amax(pValArray[:,0])
minY, maxY = np.amin(pValArray[:,1]), np.amax(pValArray[:,1])
dX = (maxX - minX)/nSubEdge
dY = (maxY - minY)/nSubEdge

xList = [[minX+nEdge*dX-nOver*dX, minX+(nEdge+1)*dX+nOver*dX] for nEdge in range(nSubEdge)]
yList = [[minY+nEdge*dY-nOver*dY, minY+(nEdge+1)*dY+nOver*dY] for nEdge in range(nSubEdge)]


sublists = []
for kx in range(nSubEdge):
    for ky in range(nSubEdge): 
        sel = (pValArray[:,0] >= xList[kx][0]) * (pValArray[:,0] <= xList[kx][1]) * (pValArray[:,1] >= yList[ky][0]) * (pValArray[:,1] <= yList[ky][1])
        sublists.append(pPosArray[sel])

knn_sub = 5
kx = 1
ky = 1
pxy = kx*nSubEdge+ky

sub = sublists[kx*nSubEdge+ky]

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import single, fcluster

y = pdist(val[sub], metric = metric)
Z = single(y)
c = fcluster(Z, 0.003, criterion='distance')
print(c)
#%%
"""
#%%
spc.norm = savu.utils.norm(spc,1,apply=False)

img = spc.shaped_array()

plt.figure()
plt.imshow(np.sum(img, axis = -1))
plt.show()


pca = savu.decomp.pca(spc, 20, center = True, returns = ['loadings','scores','predict', 'explained_variance_ratio'])

#%%


cluster = savu.cluster.kmeans(savu.utils.norm(spc,2,apply=True), 2)

img = cluster.shaped_array()

plt.figure()
plt.imshow(np.sum(img, axis = -1))
plt.show()

spc.kmeans = savu.cluster.kmeans(spc, 6)
"""
#%%
del spc[17,19:27]
del spc[17,7]
del spc[21,8]

pca = savu.decomp.pca(spc,
                      20,
                      center = False,
                      returns = ['loadings','scores','predict', 'explained_variance_ratio'],
                      norm = 2
                      )

spc.pca = pca.scores

img = pca.shaped_array()

plt.figure()
plt.imshow(img[:,:,0], cmap = 'gist_rainbow')
plt.show()

plt.figure()
plt.imshow(img[:,:,1], cmap = 'gist_rainbow')
plt.show()

plt.figure()
plt.imshow(img[:,:,2], cmap = 'gist_rainbow')
plt.show()

plt.figure()
plt.imshow(img[:,:,3], cmap = 'gist_rainbow')
plt.show()

plt.figure()
plt.plot(pca.ga('explained_variance_ratio'))
plt.show()

#%%
import numpy as np
ncomp = 2

plt.figure()
plt.imshow(np.sum(spc.shaped_array(), axis = -1))
plt.show()

a = spc.__copy__()
img = a.shaped_array()
img[img>np.quantile(img.flatten(), 0.9)] = 0

s = img.shape
x,y = np.arange(s[0]), np.arange(s[1])
x,y = np.meshgrid(x,y)
x0, y0 = s[0]//2, s[1]//2
g = 0.6
gau = lambda x, y: np.exp(- ((x-x0)**2+(y-y0)**2)/(2*g**2))
gau = gau(x,y)
gau = gau/np.sum(gau)

img2 = np.sum(img, axis = -1)
img3 = img2.__copy__()
mask = img2.__copy__()
mask[mask>0]=1

img2 = np.real(np.fft.ifft2(np.fft.fft2(img2)*np.fft.fft2(np.fft.fftshift(gau))))
mask2 = np.real(np.fft.ifft2(np.fft.fft2(mask)*np.fft.fft2(np.fft.fftshift(gau))))

img2 = img2/mask2

img3[img3==0]=img2[img3==0]

spc_new = spc[(mask>0).flatten()]
cluster = savu.cluster.kmeans(spc_new, ncomp)
c = cluster.shaped_array()
c = (np.sum(c, axis = -1)+1)*mask
plt.figure()
plt.imshow(c)
plt.show()

img_list = []
for n in range(ncomp):
    i = img3.__copy__()
    i[c!=n+1]=0
    i2 = np.real(np.fft.ifft2(np.fft.fft2(i)*np.fft.fft2(np.fft.fftshift(gau))))
    i2=i2/mask2
    i[mask==0]=i2[mask==0]
    img_list.append(i)

shape = (s[0], s[1],3)
rgb_img = np.zeros(shape, float)

rgb_list = [ [2,1,0], [0,1,2],[1,0.5,1]]

for ii, i in enumerate(img_list):
    i = np.reshape(i, (s[0], s[1],1))
    x = np.concatenate( (
        i*np.full( (s[0], s[1],1), rgb_list[ii][0])[0],
        i*np.full( (s[0], s[1],1), rgb_list[ii][1])[0],
        i*np.full( (s[0], s[1],1), rgb_list[ii][2])[0]), axis = 2 )
    
    x = (x-np.amin(x[x>0]))/(np.amax(x)-np.amin(x[x>0]))
    rgb_img = rgb_img+x
rgb_img = (rgb_img-np.amin(rgb_img))/(np.amax(rgb_img)-np.amin(rgb_img))                        
plt.figure()
plt.imshow(rgb_img)
plt.show()