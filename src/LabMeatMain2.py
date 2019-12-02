from __future__ import print_function
# from VascularGenerator import VascularGenerator
# from equations import *
# from diffu2D_u0 import lab_meat_diffuse
import os
import math
#import seaborn
import matplotlib

if os.sys.platform == "linux" or os.sys.platform == "linux2":
    matplotlib.use('TKAgg')
elif os.sys.platform == "darwin":
    matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

#import imageio
from natsort import natsorted, ns
from sklearn.preprocessing import minmax_scale

from optimizers import adamVas as AdamTwo
from optimizers import adamTest as AdamTest
from autograd.builtins import isinstance, tuple, list

#from VasGen2 import VasGen2
import random


# import numpy as np

import autograd.numpy as np
from autograd import grad
from autograd.scipy.integrate import odeint
from autograd.builtins import tuple
# from autograd.misc.optimizers import adam
import autograd.numpy.random as npr
import autograd.scipy.signal as sig
from timeit import default_timer as timer
from autograd.tracer import trace, Node

def sigmoid(x):
    # return 1 / (1 + np.exp(-x))
    return x

def jax_sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)

def gaussian(x):
    return np.exp(-x**2)


def get_submatrix_add(lst_matrix, center_pt_tuple, convolution):
    np_matrix = np.array(lst_matrix)
    r, c = np_matrix.shape
    lt_padding = 0 + (center_pt_tuple[0] - 1)
    rt_padding = (c-1) - (center_pt_tuple[0] + 1)
    top_padding = 0 + (center_pt_tuple[1] - 1)
    btm_padding = (r-1) - (center_pt_tuple[1] + 1)
    row_start = 0
    row_end = np.array(convolution).shape[0]
    col_start = 0
    col_end = np.array(convolution).shape[1]

    if lt_padding < 0:
        lt_padding = 0
        row_start = row_start + 1
    if rt_padding < 0:
        rt_padding = 0
        row_end = row_end - 1
    if top_padding < 0:
        top_padding = 0
        col_start = col_start + 1
    if btm_padding < 0:
        btm_padding = 0
        col_end = col_end - 1

    padded_convo = np.pad(np.array(convolution)[row_start:row_end, col_start:col_end], ((top_padding, btm_padding), (lt_padding, rt_padding)), mode='constant', constant_values=(0, 0))

    try:
        new_matrix = np_matrix + padded_convo
    except Exception as e:
        print(e)
        print('left pad: ', lt_padding)
        print('right pad: ', rt_padding)
        print('top pad: ', top_padding)
        print('btm pad: ', btm_padding)
        print('r start:', row_start)
        print('r end:', row_end)
        print('c start:', col_start)
        print('c end:', col_end)
        print(np.array(convolution)[row_start:row_end, col_start:col_end])
        print(np.array(convolution)[row_start:row_end, col_start:col_end].shape)

    return new_matrix

# def get_submatrix_add(lst_matrix, center_pt_tuple, convolution, submatrix_size=2):
#     print('Lst_Matrix:')
#     print(np.array(lst_matrix))
#     print('\n')
#     w, h = len(lst_matrix), len(lst_matrix[0])
#     orig_row, orig_col = center_pt_tuple
#     row, col = orig_row, orig_col
#     row_start = -1
#     row_end = -1
#     col_start = -1
#     col_end = -1

#     if row >= 0 and row < h:
#         row_start = row - submatrix_size + 1 if (row - submatrix_size + 1) > 0 else 0
#         row_end = row + submatrix_size if (row + submatrix_size) <= h else h
#     elif row == h:
#         row -= 1
#         row_start = row - submatrix_size + 1 if (row - submatrix_size + 1) > 0 else 0
#         row_end = row + submatrix_size if (row + submatrix_size) <= h else h
#     elif row < 0 or row > h:
#         print('Bad row dimension of ', row)

#     if col >= 0 and col < w:
#         col_start = col - submatrix_size + 1 if (col - submatrix_size + 1) > 0 else 0
#         col_end = col + submatrix_size if (col + submatrix_size) <= w else w
#     elif col == w:
#         col -= 1
#         col_start = col - submatrix_size + 1 if (col - submatrix_size + 1) > 0 else 0
#         col_end = col + submatrix_size if (col + submatrix_size) <= w else w
#     elif col < 0 or col > w:
#         print('Bad col dimension of ', col)

#     conv_row_start = 0
#     conv_row_end = 3
#     conv_col_start = 0
#     conv_col_end = 3
#     new_convolution = None

#     if (orig_row - submatrix_size + 1) < 0:
#         conv_row_start = 1
#     if (orig_row + submatrix_size ) >= h:
#         conv_row_end = 1

#     if (orig_col - submatrix_size + 1) < 0:
#         conv_col_start = 1
#     if (orig_col + submatrix_size) >= w:
#         conv_col_end = 1

#     new_convolution = convolution[conv_row_start:conv_row_end][conv_col_start:conv_col_end]
#     print('New Convo: ')
#     print(new_convolution)
#     print()

#     print(conv_col_start)
#     new_matrix = []
#     for row_idx, lst_row in enumerate(lst_matrix):
#         count = 0
#         if row_idx >= row_start and row_idx < row_end and count < conv_col_end:
#             val2 = new_convolution[conv_col_start + count]
#             print('    lst_row: ', lst_row[col_start:col_end])
#             updated_lst_vals = [sum(x) for x in zip(lst_row[col_start:col_end], val2)]
#             # updated_lst_vals = [sum(x) for x in zip(lst_row[row_start:row_end], val2)]
#             print('    Val2   :', val2)
#             print('    updated:', updated_lst_vals)

#             new_row=[]
#             row_count = 0
#             for cur_col_idx, val in enumerate(lst_row):
#                 if cur_col_idx >= col_start and cur_col_idx < col_end:
#                     new_row.append(updated_lst_vals[0+row_count])
#                     row_count = row_count + 1
#                 else:
#                     new_row.append(val)

#             count = count + 1
#             lst_row = new_row

#         new_matrix.append(lst_row)
#     print('New Matrix')
#     print(np.array(new_matrix))
#     print('\n')
#     return new_matrix


def getSampleParameters():
    # how many iterations to run the ODE for one trajectory, and the time step
    #     (max_T, samples per trajectory) = getSampleParameters()
    return (5, 25) #3 15 works
def loss_health(img, vesselPts, threshold, iter):
    # 2D array of neutrient values
    # sum of sigmoid values (high N is low low, low N is high loss)
    total_loss = 0.0
    for ix,iy in np.ndindex(img.shape):
        # NEED TO ONLY APPLY TO NON-VESSEL GRID LOCATIONS!!!!!!
        if not (ix, iy) in vesselPts:
            # if too low, waste production and death
            loss = 2 * hillPlusOne(img[ix,iy], 2) - 0.2
            # if loss < 1:
            #     loss = -1
            total_loss = total_loss +  (loss)
    print('LabMeatMain Line 134 LOSS:                           ', total_loss, iter)
    return total_loss


# Basic non-linear rate functions
def hillPlus(A, k, n):
    return np.power(A, n) / (np.power(A, n) + np.power(k, n))

def hillMinus(R, k, n):
    return (1 - hillPlus(R, k, n))

def hillMinusOne(R, k):
    return hillMinus(R, k, 1)

def hillPlusOne(A, k):
    return hillPlus(A, k, 1)

# def loss_health(img, vesselPts, threshold, iter):
#     # 2D array of neutrient values
#     # sum of sigmoid values (high N is low low, low N is high loss)
#     # print(img)
#     total_loss = 0.0
#     for ix,iy in np.ndindex(img.shape):
#         # NEED TO ONLY APPLY TO NON-VESSEL GRID LOCATIONS!!!!!!
#         if not (ix, iy) in vesselPts:
#             val = img[ix,iy]
#             loss = 1
#             if val >= threshold:
#                 loss = 1 / (1 + img[ix,iy])
#             # loss = gaussian(img[ix,iy]/img.mean())
#             total_loss = total_loss +  (loss)
#         # else:
#         #     print('vessel pt: ', (ix, iy))
#     print('LabMeatMain Line 171 LOSS:                           ', total_loss, iter)
#     return total_loss

def create_loss_map(img, vesselPts, threshold, iter):
    h, w = np.array(img).shape
    loss_map = []
    for _ in range(w):
        loss_map.append([0.0 for _ in range(h)])


    for ix,iy in np.ndindex(img.shape):
        # NEED TO ONLY APPLY TO NON-VESSEL GRID LOCATIONS!!!!!!
        if not (ix, iy) in vesselPts:
            # if too low, waste production and death
            loss = 2 * hillPlusOne(img[ix,iy], 2) - 0.2
            # if loss < 1:
            #     loss = 1
            loss_map[ix][iy] = loss


    # for ix,iy in np.ndindex(img.shape):
    #     # NEED TO ONLY APPLY TO NON-VESSEL GRID LOCATIONS!!!!!!
    #     if not (ix, iy) in vesselPts:
    #         val = img[ix,iy]
    #         loss = 1
    #         if val >= threshold:
    #             loss_map[ix][iy] = 1 / (1 + val)
    #         else:
    #             loss_map[ix][iy] = loss
            # loss = gaussian(img[ix,iy]/img.mean())

    # for ix,iy in np.ndindex(img.shape):
    #     # loss_map[ix][iy] = gaussian(img[ix,iy]/img.mean())
    #     loss_map[ix][iy] = 1 / (1 + img[ix,iy])
    # ideal = np.ones(img.shape)[1:,1:]
    # val = np.abs(ideal - img[1:,1:])
    return np.array(loss_map)[1:,1:]
    # return val

def initial_diffusion(mvble_pts, vesselPts, img, vesselImage):
    D = 0.1
    # D = 1
    B = D / 4

    curDiffuse = 0.0
    prevDiffuse = -1.0
    while True:
        if round(curDiffuse, 6) == round(prevDiffuse, 6):
            print('    curDiffuse: ', curDiffuse)
            print('    prevDiffuse: ', prevDiffuse)
            break
        prevDiffuse = curDiffuse
        # convolve = np.array([[1*D, 1*D, 1*D],[1*D,-8*D,1*D], [1*D, 1*D, 1*D]])
        convolve = np.array([[(1-np.sqrt(2))*D, 1*D,(1-np.sqrt(2))*D],
                             [1*D,(-8+(4*np.sqrt(2)))*D,1*D],
                             [(1-np.sqrt(2))*D, 1*D, (1-np.sqrt(2))*D]])
        deltaDiffusion = sig.convolve(np.array(img), convolve)[1:-1, 1:-1] #take off first and last
        # the update to the img from one step of diffusion
        img = np.array(np.array(img) + np.array(deltaDiffusion) + vesselImage + np.array(nonlinearDiffusion(mvble_pts, img, D, convolve)))
        # print('HERS')
        # print(np.array(img))
        # os.sys.exit()
        img = img - (B * img)
        img = np.clip(img, 0, 1e9)
        curDiffuse = np.sum(img)

    return np.array(img)

def diffusion(mvble_pts, vesselPts, img, vesselImage):
    # D is the defusion constant
    # D = .225
    # B = D / 10
    # print(np.array(img))
    # img = img[1:-1, 1:-1]
    # os.sys.exit()
    D = 0.1
    # D = 1
    B = D / 4
    # D = 0.00000000
    # B = D / 4

    #https://programtalk.com/python-examples/autograd.scipy.signal.convolve/
    for _ in range(0, 60): # how many times you run a diffusion update
        # convolve = np.array([[1*D, 1*D, 1*D],[1*D,-8*D,1*D], [1*D, 1*D, 1*D]])
        convolve = np.array([[(1-np.sqrt(2))*D, 1*D,(1-np.sqrt(2))*D],
                             [1*D,(-8+(4*np.sqrt(2)))*D,1*D],
                             [(1-np.sqrt(2))*D, 1*D, (1-np.sqrt(2))*D]])

        deltaDiffusion = sig.convolve(np.array(img), convolve)[1:-1, 1:-1] #take off first and last
        # deltaDiffusion = deltaDiffusion + np.array(img)
        # the update to the img from one step of diffusion
        img = np.array(np.array(img) + np.array(deltaDiffusion) + vesselImage + np.array(nonlinearDiffusion(mvble_pts, img, D, convolve)))
        img = img - (B * img)
        img = np.clip(img, 0, 1e9)
    #     print(type(img))
    #     img_pic = np.pad(img, ((2, 3), (2, 3)), 'constant')
    #     plt.imsave('diffusePngs/TestDiffuse_'+str(i)+'.png', np.rot90(np.array(list(img_pic))), cmap='jet')

    # path_to_img_dir = 'diffusePngs/'
    # images = []
    # for file_name in natsorted(os.listdir(path_to_img_dir), key=lambda y: y.lower()):
    #     if file_name.endswith('.png'):
    #         file_path = os.path.join(path_to_img_dir, file_name)
    #         images.append(imageio.imread(file_path))
    # imageio.mimsave('VascDiffuse.gif', images, fps=5)

    # newimg = np.array(minmax_scale(np.array(img._value)))
    # np_img = np.array(img)
    # mn, mx = np_img.min(), np_img.max()
    # np_img = (np_img - mn) / (mx - mn)
    # print(np.array(img).max(), np.array(img).min())
    print('Diffusion Nut Count: ', np.sum(img))
    return np.array(img)

def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5
    
def distToConc(d):
    #maps distance to concentration
    # distance is bound between 0.5 and math.sqrt(0.5 * 0.5 + 1.5*1.5)
    return 1 - d
    # return d - 1

# None linear diffusion (compute each convoution for each location)
def nonlinearDiffusion(mvble_pts, img, D, linearConv):
    #http://greg-ashton.physics.monash.edu/applying-python-functions-in-moving-windows.html
    #https://stackoverflow.com/questions/12816293/vectorize-this-convolution-type-loop-more-efficiently-in-numpy
    h, w = np.array(img).shape
    deltaDomain2 = []
    for _ in range(w):
        deltaDomain2.append([0.0 for _ in range(h)])
    # maximum distance from moveable point to center of neighboring grid location
    #maxD = math.sqrt(0.5 * 0.5 + 1.5*1.5)
    for i in range(len(mvble_pts)):
        pt = mvble_pts[i]
        x = pt[0]
        y = pt[1]
        int_x = 0
        int_y = 0

        # print(mvble_pts[i][0], type(mvble_pts[i][0]))
        # get the int coordinates of this moveable point
        if type(x) != type(np.array((1,1))) and type(x) != type(1) and (type(mvble_pts[i][0]) != float and type(mvble_pts[i][0]) != np.float64):
        # if type(x) != type(np.array((1,1))) and type(mvble_pts[i][0]) != np.float64:
            int_x = int(np.array(mvble_pts[i][0]._value)) 
            int_y = int(np.array(mvble_pts[i][1]._value))
        else:
            int_x = int(np.array(mvble_pts[i][0]))
            int_y = int(np.array(mvble_pts[i][1]))
        # int_x = int(np.array(mvble_pts[i][0]._value))
        # int_y = int(np.array(mvble_pts[i][1]._value))
        np_pt = np.array([x, y])

        inc = 0.5 # to center of neighbor grid locations
        # compute all the distances with the neighbors
        dist_0 = np.linalg.norm(np_pt - np.array([int_x-1 + inc, int_y-1 + inc]))
        dist_1 = np.linalg.norm(np_pt - np.array([int_x + inc, int_y-1 + inc]))
        dist_2 = np.linalg.norm(np_pt - np.array([int_x+1 + inc, int_y-1 + inc]))

        dist_3 = np.linalg.norm(np_pt - np.array([int_x-1 + inc, int_y + inc]))
        dist_4 = np.linalg.norm(np_pt - np.array([int_x + inc, int_y + inc]))
        dist_5 = np.linalg.norm(np_pt - np.array([int_x+1 + inc, int_y + inc]))

        dist_6 = np.linalg.norm(np_pt - np.array([int_x-1 + inc, int_y+1 + inc]))
        dist_7 = np.linalg.norm(np_pt - np.array([int_x + inc, int_y+1 + inc]))
        dist_8 = np.linalg.norm(np_pt - np.array([int_x+1 + inc, int_y+1 + inc]))
        # calculate non-linear concentrations adjustments needed based on the position of the point at that grid location
        # contribution of concentration to neighbor adjusts the contribution already calculated by linear diffution
        # let d be the distance from the moveable point to the center of the neighboring grid location
        # then adjustment in concentration is 1 - d.
        # if d=1, no adjustment (realdy done this)
        # if d<1, then closer so positive adjustment
        # if d>1, then farther away so negative adjustment
        # let this function be distToConc(d) --> conc
        # center = sum([distToConc(d) for d in [dist_0, dist_1, dist_2, dist_3, dist_4, dist_5, dist_6, dist_7, dist_8]]) * -1
        # convolution = [[distToConc(dist_0), distToConc(dist_1), distToConc(dist_2)], 
        #                [distToConc(dist_3), center, distToConc(dist_5)], 
        #                [distToConc(dist_6), distToConc(dist_7), distToConc(dist_8)]]
        center = sum([d for d in [dist_0, dist_1, dist_2, dist_3, dist_4, dist_5, dist_6, dist_7, dist_8]]) * -1
        convolution = [[dist_0, dist_1, dist_2], 
                       [dist_3, center, dist_5], 
                       [dist_6, dist_7, dist_8]]

        convolution = np.array(linearConv) - np.array(convolution)
        convolution = [val for val in list(convolution)]

        # center = sum([distToConc(d) for d in [dist_0, dist_1, dist_2, dist_3, dist_4, dist_5, dist_6, dist_7, dist_8]]) * -1
        # convolution = [[distToConc(dist_0), distToConc(dist_1), distToConc(dist_2)], 
        #                [distToConc(dist_3), center, distToConc(dist_5)], 
        #                [distToConc(dist_6), distToConc(dist_7), distToConc(dist_8)]]

        print('\nNonLinear Conv (After Subtracting from Linear)')
        print(np.array(convolution))
        print('Sqrt(2): ', np.sqrt(2), 1-np.sqrt(2))
        print('D0: ', dist_0) #, distToConc(dist_0))
        print('D1: ', dist_1) #, distToConc(dist_1))
        print('D2: ', dist_2) #, distToConc(dist_2))
        print('D3: ', dist_3) #, distToConc(dist_3))
        print('D4: ', sum([d for d in [dist_0, dist_1, dist_2, dist_3, dist_4, dist_5, dist_6, dist_7, dist_8]]) * -1) #, sum([distToConc(d) for d in [dist_0, dist_1, dist_2, dist_3, dist_4, dist_5, dist_6, dist_7, dist_8]]) * -1)
        print('D5: ', dist_5) #, distToConc(dist_5))
        print('D6: ', dist_6) #, distToConc(dist_6))
        print('D7: ', dist_7) #, distToConc(dist_7))
        print('D8: ', dist_8) #, distToConc(dist_8))
        # try:
        deltaDomain2 = get_submatrix_add(deltaDomain2, (int_x, int_y), convolution)
        # print(np.array(deltaDomain2))
        # print((int_x, int_y))
        # os.sys.exit()
        # except Exception as e:
        #     print(e)
        #     print(np.array(deltaDomain2), np.array(deltaDomain2).shape)
        #     print((int_x, int_y))
        #     print(np.array(convolution))
    # print(np.array(deltaDomain2) * D)
    # return np.array(deltaDomain2) * D
    return np.array(deltaDomain2)

def create_remove_imgs():
    path_to_diffuse_pngs = 'diffusePngs/'
    sim_img_folder = 'simulation_imgs/imgs/'
    sim_graph_folder = 'simulation_imgs/graphs/'
    sim_fig_folder = 'simulation_imgs/figs/'
    if not os.path.exists(path_to_diffuse_pngs):
        os.makedirs(path_to_diffuse_pngs)

    if not os.path.exists(sim_img_folder):
        os.makedirs(sim_img_folder)

    if not os.path.exists(sim_graph_folder):
        os.makedirs(sim_graph_folder)

    if not os.path.exists(sim_fig_folder):
        os.makedirs(sim_fig_folder)

    for img_file in os.listdir(path_to_diffuse_pngs):
        os.remove(path_to_diffuse_pngs + img_file)

    if os.path.exists(sim_img_folder):
        for img_file in os.listdir(sim_img_folder):
            os.remove(sim_img_folder + img_file)

    if os.path.exists(sim_graph_folder):
        for img_file in os.listdir(sim_graph_folder):
            os.remove(sim_graph_folder + img_file)

    if os.path.exists(sim_fig_folder):
        for img_file in os.listdir(sim_fig_folder):
            os.remove(sim_fig_folder + img_file)

def saveImageOne(iteration):
    #print pathOut + fileName
    # fileName = RunNAME + str(iteration).rjust(3,'0')
    fig.savefig('simulation_imgs/figs/' + str(iteration) + '.png', size=[1600,400])#, size=[1000,1000]) #, size=[700,700] IF 1000, renders each quadrant separately


def createTest(imageSize, vesselSecretion):
    # returns a simple set of parameters that can test AD
    img = np.zeros((imageSize, imageSize))
    vesselImage = np.zeros((imageSize, imageSize)) * vesselSecretion
    # movablePts = [(imageSize/2 + 0.5, imageSize - 5.5), (0.0, imageSize/2.0 )]
    movablePts = [(imageSize/2.0, imageSize/2.0)]
    vesselPts = [(int(x), int(y)) for (x, y) in movablePts]
    for (xi, yi) in vesselPts:
        vesselImage[xi, yi] = vesselSecretion

    return (img,  movablePts, vesselPts, vesselImage) 

def updateVesselImg(vesselImage, vesselSecretion, mvable_pts, img):
    vesselPts = [(int(x), int(y)) for (x, y) in mvable_pts]
    for (xi, yi) in vesselPts:
        vesselImage[xi, yi] += vesselSecretion
    img = vesselImage
    return img, vesselImage, vesselPts
    
if __name__ == "__main__":
    start = timer()
    total_iterations = 2
    imageSize = 31
    all_params = []
    all_loss = []
    vesselSecretion = 1 # how much is added each time
    # vesselSecretion = 1000 # how much is added each time
    threshold = (1 / 0.95) + 1
    create_remove_imgs()

    (max_T, count) = getSampleParameters()
    t = np.linspace(0., max_T, count)
    # (img,  movablePts, vesselPts, vesselImage) = createTest(imageSize, vesselSecretion)

    img = np.zeros((imageSize, imageSize))
    vesselImage = np.zeros((imageSize, imageSize)) * vesselSecretion
    # movablePts = [(imageSize/2, imageSize/2.0), (imageSize/2 + 0.5, imageSize - 0.5)]
    # movablePts = [(2.01,2.5)]
    movablePts = [(2.0,2.5)]
    
    # movablePts = [(imageSize/2 + 0.5, imageSize - 5.5)]
    # movablePts = [(imageSize/2.0, imageSize/2.0)]
    vesselPts = [(int(x), int(y)) for (x, y) in movablePts]
    for (xi, yi) in vesselPts:
        vesselImage[xi, yi] = vesselSecretion

    print('Starting Initial Diffusion')
    img = initial_diffusion(movablePts, vesselPts, img, vesselImage)
    print('Ended Initial Diffusion')
    plt.imshow(np.rot90(np.array(img)[1:,1:]))
    plt.show()
    # os.sys.exit()

    def fitness(movablePts, iter): 
        # vesselPts, img, vesselImage are obtained through dynamic binding
        diffused_img = diffusion(movablePts, vesselPts, img, vesselImage)
        return loss_health(diffused_img, vesselPts, threshold, iter)


    # Setup display figures
    fig = plt.figure(figsize=(16, 4), facecolor='white')
    ax_loss         = fig.add_subplot(151, frameon=True)
    ax_node_graph   = fig.add_subplot(152, frameon=True)
    ax_img          = fig.add_subplot(153, frameon=True)
    ax_diffused_img = fig.add_subplot(154, frameon=True)
    ax_loss_map     = fig.add_subplot(155, frameon=True)

    plt.show(block=False)
    count = 1
    def callback_Test(mvable_pts, iter, g):
        # img, vesselImage, vesselPts = updateVesselImg(vesselImage, vesselSecretion, mvable_pts, img)
        # Loss
        ax_loss.cla()
        ax_loss.set_title('Train Loss')
        ax_loss.set_xlabel('t')
        ax_loss.set_ylabel('loss')
        nowLoss = fitness(mvable_pts, iter)
        all_loss.append(nowLoss)
        time = np.arange(0, len(all_loss), 1)
        ax_loss.plot(time, all_loss, '-', linestyle = 'solid', label='loss') #, color = colors[i]
        ax_loss.set_xlim(time.min(), time.max())
        ax_loss.legend(loc = "upper left")

        ax_img.cla()
        ax_img.set_title('Flow Image')
        ax_img.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_img.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        imgplot = ax_img.imshow(np.rot90(np.array(img)))

        ax_diffused_img.cla()
        diffused_img_plt1 = diffusion(mvable_pts, vesselPts, img, vesselImage)
        ax_diffused_img.set_title('Diffused Image')
        ax_diffused_img.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_diffused_img.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        diffusedplot = ax_diffused_img.imshow(np.rot90(diffused_img_plt1) )

        ax_loss_map.cla()
        loss_map = create_loss_map(np.array(diffused_img_plt1), vesselPts, threshold, iter)
        ax_loss_map.set_title('Diffusion Loss Map')
        ax_loss_map.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_loss_map.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        loss_plot = ax_loss_map.imshow(np.rot90(np.array(loss_map)))

        plt.draw()
        saveImageOne(iter)
        plt.pause(0.001)
        return 3

    print('Starting AutoGrad\n')
    print('Starting mv_pts: ', movablePts)
    gradFitness = grad(fitness)
    m = np.zeros(len(movablePts))
    v = np.zeros(len(movablePts))
    step_size = 0.0002
    b1=0.9
    b2=0.999
    eps=10**-8
    
    for i in range(0, 1000):
        # best_loss = 1e9
        # best_loss_idx = -1
        # random_points = []
        # # print(list(movablePts[0]), movablePts, type(movablePts))
        # # print([random.uniform(-1, 1) + val for val in list(movablePts[0])])
        # for _ in range(8):
        #     random_points.append(tuple([random.uniform(-1, 1) + val for val in list(movablePts[0])]))
        # random_points += movablePts
        # print(random_points)

        # for idx, pt_lst in enumerate(random_points):
        #     loss = fitness([list(pt_lst)], i)
        #     if loss <= best_loss:
        #         best_loss = loss
        #         best_loss_idx = idx

        # callback_Test(movablePts, i, '')
        # movablePts = [random_points[best_loss_idx]]
        # print('\n\n')
        # print(random_points[best_loss_idx], best_loss)
        # print('\n\n')
        # img, vesselImage, vesselPts = updateVesselImg(vesselImage, vesselSecretion, movablePts, img)

        grad_pts = gradFitness(movablePts, i)
        print(grad_pts)
        callback_Test(movablePts, i, '')

        

        # m = (1 - b1) * np.array(grad_pts)      + b1 * m  # First  moment estimate.
        # v = (1 - b2) * (np.array(grad_pts)**2) + b2 * v  # Second moment estimate.
        # mhat = m / (1 - b1**(i + 1))    # Bias correction.
        # vhat = v / (1 - b2**(i + 1))

        movablePts = tuple(np.array(movablePts) + np.array(grad_pts)* .008) 
        new_pts = []
        for val in movablePts:
            x = float(val[0])
            y = float(val[1])
            if x < 0:
                x = 0.0
            elif x >= imageSize:
                x = float(imageSize -1)

            if y < 0:
                y = 0.0
            elif y >= imageSize:
                y = float(imageSize -1)
            
            new_pts += [[x, y]]
        movablePts = tuple(new_pts)

        # movablePts = movablePts + step_size * mhat / (np.sqrt(vhat) + eps)

        # os.sys.exit()
        print('Updated: point: ', movablePts)
        print(type(vesselImage), type(img))
        img, vesselImage, vesselPts = updateVesselImg(vesselImage, vesselSecretion, movablePts, img)

    # optimized_mvble_pts = AdamTest(grad(fitness), movablePts, imgsize=imageSize, step_size=0.005, num_iters=total_iterations, callback=callback_Test)
        # call display,print loss
    # optimized_mvble_pts = AdamTwo(grad(fitness), vas_structure.moveable_pts, vas_structure=vas_structure, step_size=0.005, num_iters=total_iterations, callback=callback)
    # optimized_mvble_pts = AdamTwo(grad(fitness), vas_structure.moveable_pts, vas_structure=vas_structure, step_size=0.01, num_iters=total_iterations, callback=callback)
    print('Finished AutoGrad\n')
    end = timer()
