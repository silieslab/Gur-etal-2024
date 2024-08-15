# Calculate the normalized responses of all pixel values. 

import numpy as np
import time
import sys

def Neighbours_circle(Nneigh, px_x, px_y):
    
    size_half = int(size/2)
    x_array = np.linspace(-size_half, size_half, size)
    y_array = np.linspace(-size_half, size_half, size)
    
    xx, yy = np.meshgrid(x_array, y_array)
    zz = np.sqrt((xx - px_x)**2 + (yy-px_y)**2) < Nneigh
    zz = np.reshape(zz, size*size)
    idx_zz = np.where(zz)[0]
    
    return np.array(idx_zz/size, dtype = int), idx_zz%size


N_pooling_array = int(46*0.1*int(sys.argv[1])) #np.array([ 1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80],dtype=int)) #46 pix correspond to one degree
i_image = int(sys.argv[2]) # up tp 31
set_par = 1 # 0 for woods, 1 for horizon

string = ['woods', 'horizon']

L3_image = np.loadtxt("Data/L3_responses/L3_image_responses_"+str(string[set_par])+"_"+str(i_image)+".txt")
size = L3_image.shape[0]

px_x_array, px_y_array = Neighbours_circle(N_pooling_array, 0, 0)
px_x_array -= int(size/2)
px_y_array -= int(size/2)


L3_norm = np.zeros((size, size))
L3_image_ = L3_image
counter_x = -1
for i_x in range(size):
    counter_x += 1
    counter_y = -1
    for i_y in range(size):
        counter_y += 1 
        px_x_array_ = px_x_array + counter_x
        px_y_array_ = px_x_array + counter_y
        px_x_array_ = np.where(px_x_array_<0, 0, px_x_array_)
        px_x_array_ = np.where(px_x_array_>=size, size-1, px_x_array_)
        px_y_array_ = np.where(px_y_array_<0, 0, px_y_array_)
        px_y_array_ = np.where(px_y_array_>=size, size-1, px_y_array_)
        L3_norm[i_x, i_y] = L3_image_[i_x, i_y]/(np.mean(L3_image_[px_x_array_, px_y_array_]))

np.savetxt("Data/L3_responses/L3_image_normalized_responses_"+str(string[set_par])+"_"+str(i_image)+"_pooling_"+str(N_pooling_array)+".txt", L3_norm)