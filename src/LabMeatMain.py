from __future__ import print_function
from VascularGenerator import VascularGenerator
from equations import *
from diffu2D_u0 import lab_meat_diffuse
import os
import math
import seaborn
import matplotlib

if os.sys.platform == "linux" or os.sys.platform == "linux2":
    matplotlib.use('TKAgg')
elif os.sys.platform == "darwin":
    matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

import imageio
from natsort import natsorted, ns
from sklearn.preprocessing import minmax_scale

from optimizers import adamVas as AdamTwo
from autograd.builtins import isinstance, tuple, list

from VasGen2 import VasGen2
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

def exp_decay(x):
    return np.exp(-x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    # return x

def jax_sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)

def gaussian(x):
    return np.exp(-x**2)

def get_submatrix_add(lst_matrix, center_pt_tuple, convolution, submatrix_size=2):
    w, h = len(lst_matrix), len(lst_matrix[0])
    orig_row, orig_col = center_pt_tuple
    row, col = orig_row, orig_col
    row_start = -1
    row_end = -1
    col_start = -1
    col_end = -1

    if row >= 0 and row < h:
        row_start = row - submatrix_size + 1 if (row - submatrix_size + 1) > 0 else 0
        row_end = row + submatrix_size if (row + submatrix_size) <= h else h
    elif row == h:
        row -= 1
        row_start = row - submatrix_size + 1 if (row - submatrix_size + 1) > 0 else 0
        row_end = row + submatrix_size if (row + submatrix_size) <= h else h
    elif row < 0 or row > h:
        print('Bad row dimension of ', row)

    if col >= 0 and col < w:
        col_start = col - submatrix_size + 1 if (col - submatrix_size + 1) > 0 else 0
        col_end = col + submatrix_size if (col + submatrix_size) <= w else w
    elif col == w:
        col -= 1
        col_start = col - submatrix_size + 1 if (col - submatrix_size + 1) > 0 else 0
        col_end = col + submatrix_size if (col + submatrix_size) <= w else w
    elif col < 0 or col > w:
        print('Bad col dimension of ', col)

    conv_row_start = 0
    conv_row_end = 3
    conv_col_start = 0
    conv_col_end = 3
    new_convolution = None

    if (orig_row - submatrix_size + 1) < 0:
        conv_row_start = 1
    if (orig_row + submatrix_size ) >= h:
        conv_row_end = 1

    if (orig_col - submatrix_size + 1) < 0:
        conv_col_start = 1
    if (orig_col + submatrix_size) >= w:
        conv_col_end = 1

    new_convolution = convolution[conv_row_start:conv_row_end][conv_col_start:conv_col_end]

    new_matrix = []
    for row_idx, lst_row in enumerate(lst_matrix):
        count = 0
        if row_idx >= row_start and row_idx < row_end and count < conv_col_end:
            val2 = new_convolution[conv_col_start + count]
            updated_lst_vals = [sum(x) for x in zip(lst_row[col_start:col_end], val2)]

            new_row=[]
            row_count = 0
            for cur_col_idx, val in enumerate(lst_row):
                if cur_col_idx >= col_start and cur_col_idx < col_end:
                    new_row.append(updated_lst_vals[0+row_count])
                    row_count = row_count + 1
                else:
                    new_row.append(val)

            count = count + 1
            lst_row = new_row

        new_matrix.append(lst_row)
    return new_matrix


def getSampleParameters():
    # how many iterations to run the ODE for one trajectory, and the time step
    #     (max_T, samples per trajectory) = getSampleParameters()
    return (5, 25) #3 15 works

def loss_health(img, iter):
    # np_img = np.array(img)
    # mn, mx = np_img.min(), np_img.max()
    # np_img = (np_img - mn) / (mx - mn)
    # 2D array of neutrient values
    # sum of sigmoid values (high N is low low, low N is high loss)
    total_loss = 0.0
    for ix,iy in np.ndindex(img.shape):
        loss = 1 / (1 + img[ix,iy])
        # loss = exp_decay(img[ix,iy])
        # loss = gaussian(img[ix,iy]/img.mean()
        total_loss = total_loss +  (loss)
    # ideal = np.ones(img.shape)[1:,1:]
    # total_loss = np.sum(np.abs(ideal - (img[1:,1:] /img.mean())))
    # total_loss = np.abs(ideal - (img[1:,1:])).mean()
    # total_loss = np.mean(1 / (1 + img[1:,1:])

    print('LabMeatMain Line 171 LOSS:                           ', total_loss, iter)
    return total_loss

def create_loss_map(img, iter):
    # np_img = np.array(img)
    # mn, mx = np_img.min(), np_img.max()
    # np_img = (np_img - mn) / (mx - mn)
    h, w = np.array(img).shape
    loss_map = []
    for _ in range(w):
        loss_map.append([0.0 for _ in range(h)])

    for ix,iy in np.ndindex(img.shape):
        # loss_map[ix][iy] = exp_decay(img[ix,iy])
        loss_map[ix][iy] = 1 / (1 + img[ix,iy])

    # ideal = np.ones(img.shape)[1:,1:]
    # val = np.abs(ideal - np_img[1:,1:])
    return np.array(loss_map)[1:,1:]
    # return val

def diffusion(mvble_pts, img):
    # D is the defusion constant
    # D = .225
    # B = D / 10
    # print(np.array(img))
    # img = img[1:-1, 1:-1]
    # os.sys.exit()
    D = 0.008
    B = D / 10


    # D = 0.00000000
    # B = D / 4

    #https://programtalk.com/python-examples/autograd.scipy.signal.convolve/
    for _ in range(0, 60): # how many times you run a diffusion update
        convolve = np.array([[1*D, 1*D, 1*D],[1*D,-8*D,1*D], [1*D, 1*D, 1*D]])
        deltaDiffusion = sig.convolve(np.array(img), convolve)[1:-1, 1:-1] #take off first and last
        # deltaDiffusion = deltaDiffusion + np.array(img)

        # the update to the img from one step of diffusion
        img = np.array(np.array(img) + np.array(deltaDiffusion) + np.array(nonlinearDiffusion(mvble_pts, img, D)))
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
    return np.array(img)

def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5

# None linear diffusion (compute each convoution for each location)
def nonlinearDiffusion(mvble_pts, img, D):
    #http://greg-ashton.physics.monash.edu/applying-python-functions-in-moving-windows.html
    #https://stackoverflow.com/questions/12816293/vectorize-this-convolution-type-loop-more-efficiently-in-numpy
    h, w = np.array(img).shape
    deltaDomain2 = []
    for _ in range(w):
        deltaDomain2.append([0.0 for _ in range(h)])

    for i in range(len(mvble_pts)):
        pt = mvble_pts[i]
        x = pt[0]
        y = pt[1]
        int_x = 0
        int_y = 0


        if type(x) != type(np.array((1,1))) and type(x) != type(1):
        # if type(x) != type(np.array((1,1))) and type(mvble_pts[i][0]) != np.float64:
            int_x = int(np.array(mvble_pts[i][0]._value)) 
            int_y = int(np.array(mvble_pts[i][1]._value))
        else:
            int_x = int(np.array(mvble_pts[i][0]))
            int_y = int(np.array(mvble_pts[i][1]))
        # int_x = int(np.array(mvble_pts[i][0]._value))
        # int_y = int(np.array(mvble_pts[i][1]._value))
        np_pt = np.array([x, y])

        inc = 0.5
        dist_0 = np.linalg.norm(np_pt - np.array([int_x-1 + inc, int_y-1 + inc]))
        dist_1 = np.linalg.norm(np_pt - np.array([int_x + inc, int_y-1 + inc]))
        dist_2 = np.linalg.norm(np_pt - np.array([int_x+1 + inc, int_y-1 + inc]))

        dist_3 = np.linalg.norm(np_pt - np.array([int_x-1 + inc, int_y + inc]))
        dist_4 = np.linalg.norm(np_pt - np.array([int_x + inc, int_y + inc]))
        dist_5 = np.linalg.norm(np_pt - np.array([int_x+1 + inc, int_y + inc]))

        dist_6 = np.linalg.norm(np_pt - np.array([int_x-1 + inc, int_y+1 + inc]))
        dist_7 = np.linalg.norm(np_pt - np.array([int_x + inc, int_y+1 + inc]))
        dist_8 = np.linalg.norm(np_pt - np.array([int_x+1 + inc, int_y+1 + inc]))

        # X = - sigmoid(dist_0 - 1) - sigmoid(dist_1 - 1) - sigmoid(dist_2 - 1) - sigmoid(dist_3 - 1) - sigmoid(dist_4 - 1) - sigmoid(dist_5 - 1) - sigmoid(dist_6 - 1) - sigmoid(dist_7 - 1) - sigmoid(dist_8 - 1)
        X = - exp_decay(dist_0 - 1) - exp_decay(dist_1 - 1) - exp_decay(dist_2 - 1) - exp_decay(dist_3 - 1) - exp_decay(dist_4 - 1) - exp_decay(dist_5 - 1) - exp_decay(dist_6 - 1) - exp_decay(dist_7 - 1) - exp_decay(dist_8 - 1)
        # X = - sigmoid(dist_0 - 1) - sigmoid(dist_1 - 1) - sigmoid(dist_2 - 1) - sigmoid(dist_3 - 1) - sigmoid(dist_5 - 1) - sigmoid(dist_6 - 1) - sigmoid(dist_7 - 1) - sigmoid(dist_8 - 1)
        # X = -sigmoid(dist_0 - 1 - inc) - sigmoid(dist_1 - 1 - inc) - sigmoid(dist_2 - 1 - inc) - sigmoid(dist_3 - 1 - inc) - sigmoid(dist_5 - 1 + inc) - sigmoid(dist_6 - 1 + inc) - sigmoid(dist_7 - 1 + inc) - sigmoid(dist_8 - 1 + inc)
        # X = -sigmoid(dist_0 - 1 - inc) - sigmoid(dist_1 - 1) - sigmoid(dist_2 - 1 + inc) - sigmoid(dist_3 - 1 - inc) - sigmoid(dist_5 - 1 + inc) - sigmoid(dist_6 - 1 - inc) - sigmoid(dist_7 - 1) - sigmoid(dist_8 - 1 + inc)
        # X = -sigmoid(dist_0 - 1) + inc - sigmoid(dist_1 - 1) + inc - sigmoid(dist_2 - 1) + inc - sigmoid(dist_3 - 1) + inc - sigmoid(dist_5 - 1) + inc - sigmoid(dist_6 - 1) + inc - sigmoid(dist_7 - 1) + inc - sigmoid(dist_8 - 1) + inc
        # X = -sigmoid(dist_0+ inc - 1) - sigmoid(dist_1+ inc - 1) - sigmoid(dist_2+ inc - 1) - sigmoid(dist_3 + inc- 1) - sigmoid(dist_5+ inc - 1) - sigmoid(dist_6+ inc - 1) - sigmoid(dist_7+ inc - 1) - sigmoid(dist_8+ inc - 1)
        # X = -dist_0 - dist_1 - dist_2 - dist_3 -dist_4 - dist_5 - dist_6 - dist_7 - dist_8
        # X = -dist_0 - dist_1 - dist_2 - dist_3 + dist_5 + dist_6 + dist_7 + dist_8

        # convolution = [[sigmoid(dist_0 - 1), sigmoid(dist_1 - 1), sigmoid(dist_2 - 1)], [sigmoid(dist_3 - 1), X, sigmoid(dist_5 - 1)], [sigmoid(dist_6 - 1), sigmoid(dist_7 - 1), sigmoid(dist_8 - 1)]]
        convolution = [[exp_decay(dist_0 - 1), exp_decay(dist_1 - 1), exp_decay(dist_2 - 1)], [exp_decay(dist_3 - 1), X, exp_decay(dist_5 - 1)], [exp_decay(dist_6 - 1), exp_decay(dist_7 - 1), exp_decay(dist_8 - 1)]]
        # convolution = [[sigmoid(dist_0 - 1 - inc), sigmoid(dist_1 - 1 - inc), sigmoid(dist_2 - 1 - inc)], [sigmoid(dist_3 - 1 - inc), X, sigmoid(dist_5 - 1 + inc)], [sigmoid(dist_6 - 1 + inc), sigmoid(dist_7 - 1 + inc), sigmoid(dist_8 - 1 + inc)]]
        # convolution = [[sigmoid(dist_0 - 1 - inc), sigmoid(dist_1 - 1), sigmoid(dist_2 - 1 + inc)], [sigmoid(dist_3 - 1 - inc), X, sigmoid(dist_5 - 1 + inc)], [sigmoid(dist_6 - 1 - inc), sigmoid(dist_7 - 1), sigmoid(dist_8 - 1 + inc)]]
        # convolution = [[sigmoid(dist_0 - 1) + inc, sigmoid(dist_1 - 1) + inc, sigmoid(dist_2 - 1) + inc], [sigmoid(dist_3 - 1 + inc), X + inc, sigmoid(dist_5 - 1) + inc], [sigmoid(dist_6 - 1) + inc, sigmoid(dist_7 - 1) + inc, sigmoid(dist_8 - 1) + inc]]
        # convolution = [[sigmoid(dist_0+ inc - 1), sigmoid(dist_1+ inc - 1), sigmoid(dist_2+ inc - 1)], [sigmoid(dist_3+ inc - 1), X+ inc, sigmoid(dist_5+ inc - 1)], [sigmoid(dist_6+ inc - 1), sigmoid(dist_7+ inc - 1), sigmoid(dist_8+ inc - 1)]]
        # convolution = [[-dist_0, -dist_1, -dist_2], [-dist_3, X, dist_5], [dist_6, dist_7, dist_8]]
        # convolution = [[dist_0, dist_1, dist_2], [dist_3, X, dist_5], [dist_6, dist_7, dist_8]]
        deltaDomain2 = get_submatrix_add(deltaDomain2, (int_x, int_y), convolution)

    return np.array(deltaDomain2) * D

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



if __name__ == "__main__":
    start = timer()
    total_iterations = 200
    all_params = []
    all_loss = []

    create_remove_imgs()

    (max_T, count) = getSampleParameters()
    t = np.linspace(0., max_T, count)

    print('Creating Vas')
    # vas_structure = VascularGenerator(max_range=100, num_of_nodes=2)
    # vas_structure = VasGen2(max_range=100, num_of_nodes=10)
    vas_structure = VasGen2(max_range=20, num_of_nodes=2, side_nodes=False)
    print('CreatED Vas')
    vas_structure.print_images(graph_name='AutoGrad_startGraph.png', img_name='AutoGrad_startImg.png')

    # Solve for flow
    flowDict = computeFlow(vas_structure)
    vas_structure.add_flows_to_img(flowDict)
    img = vas_structure.img
    diffused_img = img
    # vas_structure.diffused_img = img


    def fitness(fit_mvable_pts, iter):
        diffused_img = diffusion(fit_mvable_pts, img)
        # vas_structure.diffused_img = diffusion(fit_mvable_pts, img)
        # diffused_img = vas_structure.diffused_img
        # return loss_health(vas_structure.diffused_img, iter)
        return loss_health(diffused_img, iter)


    # Setup display figures
    fig = plt.figure(figsize=(16, 4), facecolor='white')
    # use LaTeX fonts in the plot
    #plt.rc('font', family='serif')
    ax_loss         = fig.add_subplot(151, frameon=True)
    ax_node_graph   = fig.add_subplot(152, frameon=True)

    ax_img          = fig.add_subplot(153, frameon=True)
    ax_diffused_img = fig.add_subplot(154, frameon=True)
    ax_loss_map     = fig.add_subplot(155, frameon=True)

    plt.show(block=False)
    count = 1
    imgcolorbar = None
    # Plot Data through callback
    def callback(mvable_pts, iter, g):
        # imgcolorbar.remove()
        # diffusedcolorbar.remove()
        # losscolorbar.remove()
        # print('callback', mvable_pts, iter)
        ####################################
        # LOSS as a function of TIME
        ax_loss.cla()
        ax_loss.set_title('Train Loss')
        ax_loss.set_xlabel('t')
        ax_loss.set_ylabel('loss')
        #colors =['b', 'b', 'g', 'g', 'r', 'r']
        nowLoss = fitness(mvable_pts, iter)
        # flowDict = computeFlow(vas_structure)
        # vas_structure.add_flows_to_img(flowDict)
        # nowLoss = loss_health(vas_structure.diffused_img, iter)
        # nowLoss = loss_health(np.array(diffused_img), iter)
        # print('NowLoss: ', nowLoss, type(nowLoss))
        all_loss.append(nowLoss)
        time = np.arange(0, len(all_loss), 1)
        
        # print(all_loss, type(all_loss))
        ax_loss.plot(time, all_loss, '-', linestyle = 'solid', label='loss') #, color = colors[i]
        ax_loss.set_xlim(time.min(), time.max())
        ax_loss.legend(loc = "upper left")

        ## Plots the Node Graph
        ax_node_graph.cla()
        ax_node_graph.set_title('Node Graph')
        for j, s in enumerate(vas_structure.tri.simplices):
            p = np.array(vas_structure.pts)[s].mean(axis=0)
            ax_node_graph.text(p[0], p[1], 'Cell #%d' % j, ha='center') # label triangles
        ax_node_graph.triplot(np.array(vas_structure.pts)[:,0], np.array(vas_structure.pts)[:,1], vas_structure.tri.simplices)
        ax_node_graph.plot(np.array(vas_structure.pts)[:,0], np.array(vas_structure.pts)[:,1], 'o')

        # ==== Plot Img Version ==== #
        # pltimg = np.pad(np.array(vas_structure.img), ((2, 3), (2, 3)), 'constant')
        # plt.imsave(img_name, np.rot90(pltimg), cmap='jet')
        ax_img.cla()
        imgcolorbar = None
        ax_img.set_title('Flow Image')
        ax_img.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_img.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        # ax_img.imshow(np.rot90(pltimg))
        imgplot = ax_img.imshow(np.rot90(np.array(vas_structure.img)[1:,1:]))
        # imgcolorbar = fig.colorbar(mappable=imgplot, ax=ax_img, orientation='horizontal')

        # ==== Plot Diffused Img Version ==== #
        ax_diffused_img.cla()
        diffused_img_plt1 = diffusion(mvable_pts, vas_structure.img)
        # diffused_img_plt1 = np.pad(np.array(diffused_img_plt1), ((2, 3), (2, 3)), 'constant')
        # diffused_img_plt1 = np.pad(np.array(vas_structure.diffused_img), ((2, 3), (2, 3)), 'constant')
        ax_diffused_img.set_title('Diffused Image')
        ax_diffused_img.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_diffused_img.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        # print(np.array(vas_structure.diffused_img))
        diffusedplot = ax_diffused_img.imshow(np.rot90(diffused_img_plt1[1:,1:]) )
        # diffusedcolorbar = fig.colorbar(mappable=diffusedplot, ax=ax_diffused_img, orientation='horizontal')

        # ax_diffused_img.imshow(np.rot90(np.array(vas_structure.diffused_img)._value))

        # ==== Plot Diffused LOSS MAP ==== #
        ax_loss_map.cla()
        # loss_map = create_loss_map(vas_structure.diffused_img, iter)
        loss_map = create_loss_map(np.array(diffused_img_plt1), iter)
        # loss_map_plt1 = np.pad(np.array(loss_map), ((2, 3), (2, 3)), 'constant')
        ax_loss_map.set_title('Diffusion Loss Map')
        ax_loss_map.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_loss_map.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        loss_plot = ax_loss_map.imshow(np.rot90(np.array(loss_map)))
        # losscolorbar = fig.colorbar(mappable=loss_plot, ax=ax_loss_map, orientation='horizontal')
        # losscolorbar.set_clim(np.rot90(np.array(loss_map)).min(), np.rot90(np.array(loss_map).max()))



        plt.draw()
        saveImageOne(iter)
        # imgcolorbar.remove()
        # diffusedcolorbar.remove()
        # losscolorbar.remove()
        plt.pause(0.001)
        return 3

    pts = np.array(vas_structure.pts)
    print('Starting AutoGrad\n')
    print('Original MvPts: ', vas_structure.moveable_pts)
    optimized_mvble_pts = AdamTwo(grad(fitness), vas_structure.moveable_pts, vas_structure=vas_structure, step_size=0.3, num_iters=total_iterations, callback=callback)
    # optimized_mvble_pts = AdamTwo(grad(fitness), vas_structure.moveable_pts, vas_structure=vas_structure, step_size=0.005, num_iters=total_iterations, callback=callback)
    # optimized_mvble_pts = AdamTwo(grad(fitness), vas_structure.moveable_pts, vas_structure=vas_structure, step_size=0.01, num_iters=total_iterations, callback=callback)
    print('Finished AutoGrad\n')

    print('    Optimized Pts:')
    print('      ', optimized_mvble_pts)
    print('\n')

    # print('Optimized PTs')
    # print(vas_structure.pts)
    # vas_structure.update_moveable_pts(optimized_mvble_pts)
    print('Setting up optimized points')
    print(np.array(vas_structure.pts))
    # print()
    # print(pts)
    # vas_structure.print_images(graph_name='Optimized_AutoGrad_graph.png', img_name='Optimized_AutoGrad_img.png')
    end = timer()
    print('Time per iteration: ', str((end-start) / total_iterations))

    def img_path_generator(path_to_img_dir):
        for file_name in natsorted(os.listdir(path_to_img_dir), key=lambda y: y.lower()):
            if file_name.endswith('.png'):
                file_path = os.path.join(path_to_img_dir, file_name)
                yield imageio.imread(file_path)

    # sim_img_folder = 'simulation_imgs/imgs/'
    # sim_graph_folder = 'simulation_imgs/graphs/'
    sim_fig_folder = 'simulation_imgs/figs/'

    # imageio.mimsave('AutoDiff_Img.gif', img_path_generator(sim_img_folder), fps=50)
    # imageio.mimsave('AutoDiff_Graph.gif', img_path_generator(sim_graph_folder), fps=50)
    imageio.mimsave('AutoDiff_Figs.gif', img_path_generator(sim_fig_folder), fps=50)
