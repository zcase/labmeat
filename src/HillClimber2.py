from VascularGenerator import VascularGenerator
from VasGen2 import VasGen2
from equations import *
from diffu2D_u0 import lab_meat_diffuse
import os
import math
import matplotlib.pyplot as plt

import random
import time

import imageio
from natsort import natsorted, ns
from sklearn.preprocessing import minmax_scale

import autograd.numpy as np
from autograd import grad
from autograd.scipy.integrate import odeint
from autograd.builtins import tuple
from autograd.misc.optimizers import adam
import autograd.numpy.random as npr
import autograd.scipy.signal as sig
from timeit import default_timer as timer
from meatModel2d import getDynamics, getTrueParameters, getSampleParameters


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def gaussian(x):
    return math.exp(-x**2)

def create_remove_imgs():
    path_to_diffuse_pngs = 'Hillclimb/diffusePngs/'
    sim_img_folder = 'Hillclimb/imgs/'
    sim_graph_folder = 'Hillclimb/graphs/'
    sim_fig_folder = 'Hillclimb/figs/'
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

# def get_submatrix_add(lst_matrix, center_pt_tuple, convolution, submatrix_size=2):
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

#     new_matrix = []
#     for row_idx, lst_row in enumerate(lst_matrix):
#         count = 0
#         if row_idx >= row_start and row_idx < row_end and count < conv_col_end:
#             val2 = new_convolution[conv_col_start + count]
#             updated_lst_vals = [sum(x) for x in zip(lst_row[col_start:col_end], val2)]

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
#     return new_matrix


# def diffusion(mvble_pts, img):
#     # D is the defusion constant
#     # D = .225
#     # B = D / 10
#     D = 0.03
#     B = D / 4
#     # D = 0.00000001
#     # B = D / 4

#     #https://programtalk.com/python-examples/autograd.scipy.signal.convolve/
#     for _ in range(0, 60): # how many times you run a diffusion update
#         convolve = np.array([[1*D, 1*D, 1*D],[1*D,-8*D,1*D], [1*D, 1*D, 1*D]])
#         deltaDiffusion = sig.convolve(np.array(img), convolve)[1:-1, 1:-1] #take off first and last
#         # deltaDiffusion = deltaDiffusion + np.array(img)

#         # the update to the img from one step of diffusion
#         img = np.array(np.array(img) + np.array(deltaDiffusion) + np.array(nonlinearDiffusion(mvble_pts, img)))
#         img = img - (B * img)
   
#     np_img = np.array(img)
#     mn, mx = np_img.min(), np_img.max()
#     np_img = (np_img - mn) / (mx - mn)
#     return np_img

# # None linear diffusion (compute each convoution for each location)
# def nonlinearDiffusion(mvble_pts, img):
#     #http://greg-ashton.physics.monash.edu/applying-python-functions-in-moving-windows.html
#     #https://stackoverflow.com/questions/12816293/vectorize-this-convolution-type-loop-more-efficiently-in-numpy
#     h, w = np.array(img).shape
#     deltaDomain2 = []
#     for _ in range(w):
#         deltaDomain2.append([0.0 for _ in range(h)])

#     for i in range(len(mvble_pts)):
#         pt = mvble_pts[i]
#         x = pt[0]
#         y = pt[1]
#         int_x = 0
#         int_y = 0


#         if type(x) != type(np.array((1,1))) and type(x) != type(1):
#             int_x = int(np.array(mvble_pts[i][0])) 
#             int_y = int(np.array(mvble_pts[i][1]))
#         else:
#             int_x = int(np.array(mvble_pts[i][0]))
#             int_y = int(np.array(mvble_pts[i][1]))
        
#         np_pt = np.array([x, y])

#         inc = 0.5
#         dist_0 = np.linalg.norm(np_pt - np.array([int_x-1 + inc, int_y-1 + inc]))
#         dist_1 = np.linalg.norm(np_pt - np.array([int_x + inc, int_y-1 + inc]))
#         dist_2 = np.linalg.norm(np_pt - np.array([int_x+1 + inc, int_y-1 + inc]))

#         dist_3 = np.linalg.norm(np_pt - np.array([int_x-1 + inc, int_y + inc]))
#         dist_4 = np.linalg.norm(np_pt - np.array([int_x + inc, int_y + inc]))
#         dist_5 = np.linalg.norm(np_pt - np.array([int_x+1 + inc, int_y + inc]))

#         dist_6 = np.linalg.norm(np_pt - np.array([int_x-1 + inc, int_y+1 + inc]))
#         dist_7 = np.linalg.norm(np_pt - np.array([int_x + inc, int_y+1 + inc]))
#         dist_8 = np.linalg.norm(np_pt - np.array([int_x+1 + inc, int_y+1 + inc]))

#         X = -dist_0 - dist_1 - dist_2 - dist_3 - dist_5 - dist_6 - dist_7 - dist_8
        
#         convolution = [[sigmoid(dist_0 - 1), sigmoid(dist_1 - 1), sigmoid(dist_2 - 1)], [sigmoid(dist_3 - 1), X, sigmoid(dist_5 - 1)], [sigmoid(dist_6 - 1), sigmoid(dist_7 - 1), sigmoid(dist_8 - 1)]]
        
#         deltaDomain2 = get_submatrix_add(deltaDomain2, (int_x, int_y), convolution)

#     return deltaDomain2

# def create_loss_map(img, iter):
#     h, w = np.array(img).shape
#     loss_map = []
#     for _ in range(w):
#         loss_map.append([0.0 for _ in range(h)])

#     ideal = np.ones(img.shape)[1:,1:]
#     val = np.abs(ideal - img[1:,1:])
#     return val

# def loss_health(img, iter, vessel_points):
#     threshold = 0.9
#     # 2D array of neutrient values
#     # sum of sigmoid values (high N is low low, low N is high loss)
#     total_loss = 0.0
#     for ix,iy in np.ndindex(img.shape):
#         if vessel_points[(ix,iy)] > 0.0:
#             val = img[ix,iy]
#             loss = 1
#             if val >= threshold:
#                 loss = 1/(1+img[ix,iy])
#             total_loss += loss
    
#     return total_loss

def adjustMoveRate(num):
    num = num/1.5
    if num < 0.5:
        if num > 0:
            num = 0.5
        elif num > -0.5:
            num = -0.5
    return num

def saveImageOne(iteration):
    fig.savefig('HillClimb/figs/' + str(iteration) + '.png', size=[1600,400])#, size=[1000,1000]) #, size=[700,700] IF 1000, renders each quadrant separately

if __name__ == "__main__":
    print("Climbing the hill")
    numNodes = 4
    path_to_diffuse_pngs = 'Hillclimb/diffusePngs/'
    sim_img_folder = 'Hillclimb/imgs/'
    sim_graph_folder = 'Hillclimb/graphs/'

    create_remove_imgs()

    vas_structure = VasGen2(max_range=20, num_of_nodes=numNodes, side_nodes=False)
    vas_structure.print_images(graph_name='HillClimb/HillClimb_startGraph.png', img_name='HillClimb/HillClimb_startImg.png')

    test_movement = np.array([-5, -2, -1, 0, 1, 2, 5])

    # def fitness(fit_mvable_pts, iter):
    #     diffused_img = diffusion(fit_mvable_pts, img)
    #     return loss_health(diffused_img, iter, vas_structure.pts_on_vessels)

    flowDict = computeFlow(vas_structure)
    vas_structure.add_flows_to_img(flowDict)
    img = vas_structure.img
    diffused_img = img
    
    img = np.array(img)
    vas_structure.Q = img
    vas_structure.product_values = np.array(np.zeros(img.shape))
    vas_structure.nutrient_values = np.array(np.zeros(img.shape))

    mvable_pts = vas_structure.moveable_pts
    
    all_loss = []
    time_lst = []

    # Set up figures
    fig = plt.figure(figsize=(16, 6), facecolor='white')
    ax_loss         = fig.add_subplot(231, frameon=True)
    ax_cpu          = fig.add_subplot(232, frameon=True)
    ax_node_graph   = fig.add_subplot(233, frameon=True)
    ax_img          = fig.add_subplot(234, frameon=True)
    ax_product      = fig.add_subplot(235, frameon=True)
    ax_nutrient     = fig.add_subplot(236, frameon=True)
    plt.show(block=False)

    def callback(mvable_pts, iter, nowLoss, time_duration):
        # ==== LOSS as a function of TIME ==== #
        ax_loss.cla()
        ax_loss.set_title('Optimization Gain')
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Fitness')
        nowLoss = nowLoss
        all_loss.append(nowLoss)
        iteration = np.arange(0, len(all_loss), 1)

        ax_loss.plot(iteration, all_loss, '-', linestyle = 'solid', label='loss') #, color = colors[i]
        ax_loss.set_xlim(iteration.min(), iteration.max())
        ax_loss.legend(loc = "upper left")

        # ==== CPU Time ==== #
        ax_cpu.cla()
        ax_cpu.set_title('Fitness vs CPU Time')
        ax_cpu.set_xlabel('CPU TIME')
        ax_cpu.set_ylabel('Fitness')
        time_lst.append(time_duration)
        ax_cpu.scatter(time_lst, all_loss)
        ax_cpu.legend(loc = "upper left")

        # ==== Plots the Node Graph ==== #
        ax_node_graph.cla()
        ax_node_graph.set_title('Node Graph')
        for j, s in enumerate(vas_structure.tri.simplices):
            p = np.array(vas_structure.pts)[s].mean(axis=0)
            ax_node_graph.text(p[0], p[1], 'Cell #%d' % j, ha='center') # label triangles
        ax_node_graph.triplot(np.array(vas_structure.pts)[:,0], np.array(vas_structure.pts)[:,1], vas_structure.tri.simplices)
        ax_node_graph.plot(np.array(vas_structure.pts)[:,0], np.array(vas_structure.pts)[:,1], 'o')

        # ==== Plot Flow Img ==== #
        ax_img.cla()
        ax_img.set_title('Flow Image')
        ax_img.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_img.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax_img.imshow(np.rot90(np.array(vas_structure.img)[1:,1:]), cmap='jet')

        # ==== Plot Product Image ==== #
        ax_product.cla()
        ax_product.set_title('Product Image')
        ax_product.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_product.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax_product.imshow(np.rot90(np.array(vas_structure.product_values)), cmap='jet')
        
        # # ==== Plot Nutrient Image ==== #
        ax_nutrient.cla()
        ax_nutrient.set_title('Nutrient Image')
        ax_nutrient.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_nutrient.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax_nutrient.imshow(np.rot90(np.array(vas_structure.nutrient_values)), cmap='jet')

        plt.subplots_adjust( wspace = 0.5, hspace = 0.5 )
        plt.draw()
        saveImageOne(iter)
        plt.pause(0.001)
        return 3

    dImproved = False
    # currentLoss = fitness(mvable_pts, 0)
    currentLoss = -1
    i = 0
    timesNotImproved = 0
    while timesNotImproved < 200:
        start = time.time()

        if not dImproved:
            index = random.randrange(0, numNodes)
            inc_index_x = np.random.choice(test_movement, 1, replace=False)[0]
            inc_index_y = np.random.choice(test_movement, 1, replace=False)[0]
        else:
            inc_index_x = adjustMoveRate(inc_index_x)
            inc_index_y = adjustMoveRate(inc_index_y)

        originalPoints = vas_structure.moveable_pts[index]
            
        test_x = vas_structure.moveable_pts[index][0] + inc_index_x
        test_y = vas_structure.moveable_pts[index][1] + inc_index_y

        if test_x < 1:
            test_x = 1
        if test_x > 19:
            test_x = 19

        if test_y < 1:
            test_y = 1
        if test_y > 19:
            test_y = 19

        vas_structure.moveable_pts[index] = [test_x, test_y]
        mvable_pts = vas_structure.moveable_pts
        vas_structure.update_hillclimb_pts(mvable_pts)
        flowDict = computeFlow(vas_structure)
        vas_structure.add_flows_to_img(flowDict)
        
        
        (dynamicsTrue, fitnessList, odeDeltaList, pdeDeltaList) = getDynamics(vas_structure, 
                        getTrueParameters(), 
                        nonLinear = False, 
                        movablePts = vas_structure.moveable_pts,
                        runParameters = getSampleParameters())

        (max_t, count) = getSampleParameters()
        loss = np.cumsum(fitnessList[int(count*.30):-1])[-1]#fitness(mvable_pts, i)
        
        print('Point change:', originalPoints, mvable_pts[index])
        if loss > currentLoss:
            currentLoss = loss
            dImproved = True
            print(i, 'REDUCED LOSS:', loss)
            timesNotImproved = 0
        else:
            vas_structure.moveable_pts[index] = originalPoints
            mvable_pts = vas_structure.moveable_pts
            dImproved = False
            print(i, 'LOSS IS >= CURRENT', loss)
            timesNotImproved += 1

        end = time.time()
        elapsedTime = end-start
        callback(mvable_pts, i, currentLoss, elapsedTime)
        
        vas_structure.print_images(graph_name=sim_graph_folder+'test_graph'+str(i)+'.png', img_name=sim_img_folder+'test_img'+str(i)+'.png')
        i += 1



    def img_path_generator(path_to_img_dir):
        for i, file_name in enumerate(natsorted(os.listdir(path_to_img_dir), key=lambda y: y.lower())):
            if file_name.endswith('.png'):
                if not i == 0:
                    file_path = os.path.join(path_to_img_dir, file_name)
                    yield imageio.imread(file_path)

    sim_fig_folder = 'HillClimb/figs/'
    imageio.mimsave('HillClimb/HillClimb_Figs.gif', img_path_generator(sim_fig_folder), fps=10)

