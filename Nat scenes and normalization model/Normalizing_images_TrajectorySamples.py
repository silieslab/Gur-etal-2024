# Calculate the normalized responses of the pixel values corresponding to the trajectories simulated. 

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

if set_par == 0:
    L3_image = np.loadtxt("Data/L3_responses/L3_image_responses_"+str(string[set_par])+"_"+str(i_image)+".txt")
    size = L3_image.shape[0]
else:
    idx_y_ = 48
    idx_x_ = 400
    idx_y_max_ = 600 + 2*idx_y_
    size_new = idx_y_max_ - idx_y_
    
    if i_image == 25:
        idx_y = 48
        idx_x = idx_x_
        idx_y_max = size_new + (idx_y - idx_y_)

    if i_image == 22:
        idx_y = 75
        idx_x = idx_x_
        idx_y_max = size_new + (idx_y - idx_y_)

    if i_image == 7:
        idx_y = 0
        idx_x = idx_x_
        idx_y_max = size_new + (idx_y - idx_y_)

    if i_image == 15:
        idx_y = 0
        idx_x = idx_x_
        idx_y_max = size_new + (idx_y - idx_y_)

    if i_image == 13:
        idx_y = 90
        idx_x = idx_x_
        idx_y_max = size_new + (idx_y - idx_y_)

    L3_image = np.loadtxt("Data/L3_responses/L3_image_responses_"+str(string[set_par])+"_"+str(i_image)+".txt")[idx_x:,idx_y:idx_y_max]
    size = L3_image.shape[0]

px_x_array, px_y_array = Neighbours_circle(N_pooling_array, 0, 0)
px_x_array -= int(size/2)
px_y_array -= int(size/2)

if set_par == 0:

    Tr_type = 1
    T_x__ = np.loadtxt("Data/Data_trajectories/T_X.txt")
    if Tr_type == 1:
        T_y__ = -np.loadtxt("Data/Data_trajectories/T_Y.txt")
    else:
        T_y__ = np.loadtxt("Data/Data_trajectories/T_Y.txt")
    T_x_ = T_y__ + int(size/2)
    T_y_ = T_x__ + int(size/2)
else:
    
    Tr_type = 1
    T_x__ = np.loadtxt("Data/Data_trajectories/T_X_horizon.txt")
    T_y__ = np.loadtxt("Data/Data_trajectories/T_Y_horizon.txt")
    
    if Tr_type == 1:
        T_y_ = T_x__ + int(size/2)
        T_x_ = - T_y__ + int(size/2) + 200
    else:
        T_y_ = T_y__ + int(size/2) - 220
        T_x_ = T_x__ + int(size/2) 
    
        
N_Traj, N_SamplePoints = T_x_.shape

L3_norm = np.zeros((N_Traj, N_SamplePoints))

L3_image_ = L3_image

counter_x = -1
for i_traj in range(N_Traj): 
    counter_x += 1
    counter_y = -1
    for i_Sample in range(N_SamplePoints):
        counter_y += 1 
        
        i_x = int(T_x_[counter_x, counter_y])
        i_y = int(T_y_[counter_x, counter_y])
        
        px_x_array_ = px_x_array + i_x
        px_y_array_ = px_y_array + i_y
        px_x_array_ = np.where(px_x_array_<0, 0, px_x_array_)
        px_x_array_ = np.where(px_x_array_>=size, size-1, px_x_array_)
        px_y_array_ = np.where(px_y_array_<0, 0, px_y_array_)
        px_y_array_ = np.where(px_y_array_>=size, size-1, px_y_array_)
        L3_norm[counter_x, counter_y] = L3_image_[i_x, i_y]/(np.mean(L3_image_[px_x_array_, px_y_array_]))

np.savetxt("Data/L3_responses/L3_image_normalized_responses_"+str(string[set_par])+"_"+str(i_image)+"_pooling_"+str(N_pooling_array)+"_trajectories_"+str(Tr_type)+".txt", L3_norm)
